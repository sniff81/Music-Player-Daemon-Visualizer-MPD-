#define _GNU_SOURCE
#define _XOPEN_SOURCE_EXTENDED 1 
#define NCURSES_WIDECHAR 1


### COMPILE: ###
### gcc -O2 viz.c -o viz   $(pkg-config --cflags --libs libpipewire-0.3 ncursesw)   -lfftw3 -lm -lpthread



#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "wchar.h"
#include "math.h"
#include "locale.h"
#include "ncurses.h"
#include "fftw3.h"
#include "pthread.h"
#include "unistd.h"
#include "time.h"

#include "pipewire/pipewire.h"
#include "spa/param/audio/format-utils.h"
#include "spa/param/audio/raw.h"

#define BAR_GAP 1            // Space gap between columns
#define BAR_WIDTH 2          // Characters per bar width
#define BAR_LENGTH_SCALE 1.0f // Multiplier to adjust overall bar height/length
#define SAMPLE_RATE 44100
#define BUFFER_SIZE 2048     // ~21.53Hz per FFT bin
#define RING_SIZE 65536      // Sample ring buffer capacity

#define GRAD_STEPS 16       // 16 steps top + 16 steps bottom = 32 distinct colors
#define MAX_COLORS (2 * GRAD_STEPS)

// Threading & Ring Buffer state
volatile int keep_running = 1;

typedef struct {
    int16_t data[RING_SIZE];
    size_t write_pos;
    size_t read_pos;
    size_t count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} audio_ring_buffer_t;

struct pw_app {
    struct pw_thread_loop *thread_loop;
    struct pw_context *context;
    struct pw_core *core;
    struct pw_stream *stream;
    struct spa_hook stream_listener;
    audio_ring_buffer_t ring_buf;
};

void ring_buffer_init(audio_ring_buffer_t *rb) {
    rb->write_pos = 0;
    rb->read_pos = 0;
    rb->count = 0;
    pthread_mutex_init(&rb->mutex, NULL);
    pthread_cond_init(&rb->cond, NULL);
}

void ring_buffer_free(audio_ring_buffer_t *rb) {
    pthread_mutex_destroy(&rb->mutex);
    pthread_cond_destroy(&rb->cond);
}

void ring_buffer_write(audio_ring_buffer_t *rb, const int16_t *in, size_t count) {
    pthread_mutex_lock(&rb->mutex);
    for (size_t i = 0; i < count; i++) {
        if (rb->count >= RING_SIZE) {
            rb->read_pos = (rb->read_pos + 1) % RING_SIZE;
            rb->count--;
        }
        rb->data[rb->write_pos] = in[i];
        rb->write_pos = (rb->write_pos + 1) % RING_SIZE;
        rb->count++;
    }
    pthread_cond_signal(&rb->cond);
    pthread_mutex_unlock(&rb->mutex);
}

int ring_buffer_read(audio_ring_buffer_t *rb, int16_t *out, size_t count) {
    pthread_mutex_lock(&rb->mutex);

    while (rb->count < count && keep_running) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100000000; // 100ms timeout
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }

        if (pthread_cond_timedwait(&rb->cond, &rb->mutex, &ts) != 0) {
            break;
        }
    }

    if (!keep_running) {
        pthread_mutex_unlock(&rb->mutex);
        return -1;
    }

    size_t available = (rb->count < count) ? rb->count : count;
    for (size_t i = 0; i < available; i++) {
        out[i] = rb->data[rb->read_pos];
        rb->read_pos = (rb->read_pos + 1) % RING_SIZE;
    }
    rb->count -= available;

    if (available < count) {
        memset(out + available, 0, (count - available) * sizeof(int16_t));
    }

    pthread_mutex_unlock(&rb->mutex);
    return 0;
}

static void on_process(void *userdata) {
    struct pw_app *app = userdata;
    struct pw_buffer *b = pw_stream_dequeue_buffer(app->stream);
    if (!b) return;

    struct spa_buffer *buf = b->buffer;
    if (buf->datas[0].data != NULL) {
        const int16_t *samples = (const int16_t *)buf->datas[0].data;
        uint32_t offset = buf->datas[0].chunk->offset;
        uint32_t size = buf->datas[0].chunk->size;
        const int16_t *src = (const int16_t *)((const uint8_t *)samples + offset);
        uint32_t num_samples = size / sizeof(int16_t);

        if (num_samples > 0) {
            ring_buffer_write(&app->ring_buf, src, num_samples);
        }
    }

    pw_stream_queue_buffer(app->stream, b);
}

void init_bands(int num_bars, int (*band_ranges)[2], float *band_boost) {
    int half = num_bars / 2;
    int min_bin = 1;   // ~21.5 Hz
    int max_bin = 700; // ~15 kHz

    double log_min = log((double)min_bin);
    double log_max = log((double)max_bin);

    for (int i = 0; i < half; i++) {
        double start_f = exp(log_min + (log_max - log_min) * ((double)i / half));
        double end_f   = exp(log_min + (log_max - log_min) * ((double)(i + 1) / half));

        int b_start = (int)start_f;
        int b_end   = (int)end_f;
        if (b_end <= b_start) b_end = b_start;

        band_ranges[i][0] = b_start;
        band_ranges[i][1] = b_end;

        int mirror_idx = num_bars - 1 - i;
        band_ranges[mirror_idx][0] = b_start;
        band_ranges[mirror_idx][1] = b_end;

        float boost_val = 1.0f + 2.5f * ((float)i / (float)(half > 1 ? half - 1 : 1));
        band_boost[i] = boost_val;
        band_boost[mirror_idx] = boost_val;
    }

    if (num_bars % 2 != 0) {
        band_ranges[half][0] = max_bin / 2;
        band_ranges[half][1] = max_bin;
        band_boost[half] = 3.5f;
    }
}

char now_playing[256] = "";
pthread_mutex_t np_mutex = PTHREAD_MUTEX_INITIALIZER;

void* now_playing_worker(void* arg) {
    const char* cmd = "mpc current 2>/dev/null";

    while (keep_running) {
        FILE *fp = popen(cmd, "r");
        char buf[256] = {0};
        if (fp) {
            if (fgets(buf, sizeof(buf) - 1, fp) != NULL) {
                buf[strcspn(buf, "\r\n")] = 0;
            }
            pclose(fp);
        }

        pthread_mutex_lock(&np_mutex);
        strncpy(now_playing, buf, sizeof(now_playing) - 1);
        now_playing[sizeof(now_playing) - 1] = '\0';
        pthread_mutex_unlock(&np_mutex);

        for (int i = 0; i < 5 && keep_running; i++) {
            usleep(100000);
        }
    }
    return NULL;
}

// Dynamic pair lookup grid & state
int fg_bg_pair_grid[MAX_COLORS][MAX_COLORS + 1];
int current_pair_count = 1;

void init_gradient_colors() {
    start_color();
    use_default_colors();

    memset(fg_bg_pair_grid, 0, sizeof(fg_bg_pair_grid));
    current_pair_count = 1;

    static const float palette[5][3] = {
        {553.0f,  894.0f, 110.0f}, // YellowGreen
        {1000.0f, 1000.0f,   0.0f}, // Yellow
        {1000.0f,  647.0f,   0.0f}, // Orange
        {1000.0f,  271.0f,   0.0f}, // Orange-Red
        {1000.0f,    0.0f,   0.0f}  // Bright Red
    };

    if (has_colors() && can_change_color()) {
        for (int pos = 0; pos < GRAD_STEPS; pos++) {
            float t = (float)pos / (float)(GRAD_STEPS - 1);
            float scaled = t * 4.0f;
            int idx = (int)scaled;
            if (idx >= 4) idx = 3;
            float local_t = scaled - idx;

            int r = (int)((1.0f - local_t) * palette[idx][0] + local_t * palette[idx + 1][0]);
            int g = (int)((1.0f - local_t) * palette[idx][1] + local_t * palette[idx + 1][1]);
            int b = (int)((1.0f - local_t) * palette[idx][2] + local_t * palette[idx + 1][2]);

            // Top gradient colors: IDs 16 .. 16 + GRAD_STEPS - 1
            init_color(16 + pos, r, g, b);

            // Bottom gradient colors (30% down to 0% lightness): IDs 16 + GRAD_STEPS .. 16 + 2*GRAD_STEPS - 1
            float bot_light = 0.30f * (1.0f - t);
            init_color(16 + GRAD_STEPS + pos, (int)(r * bot_light), (int)(g * bot_light), (int)(b * bot_light));
        }
    }
}

int get_pair(int fg, int bg) {
    if (fg < 0 || fg >= MAX_COLORS) return 0;
    int bg_idx = (bg < 0 || bg >= MAX_COLORS) ? MAX_COLORS : bg;

    if (fg_bg_pair_grid[fg][bg_idx] != 0) {
        return fg_bg_pair_grid[fg][bg_idx];
    }

    if (current_pair_count < COLOR_PAIRS && current_pair_count < 32767) {
        short fg_color = (has_colors() && can_change_color()) ? (16 + fg) : COLOR_RED;
        short bg_color = (bg < 0 || bg >= MAX_COLORS) ? -1 : ((has_colors() && can_change_color()) ? (16 + bg) : COLOR_BLACK);

        init_pair(current_pair_count, fg_color, bg_color);
        fg_bg_pair_grid[fg][bg_idx] = current_pair_count;
        return current_pair_count++;
    }

    if (bg_idx != MAX_COLORS && fg_bg_pair_grid[fg][MAX_COLORS] != 0) {
        return fg_bg_pair_grid[fg][MAX_COLORS];
    }

    return 0;
}

int get_gradient_color_idx(float pos_ratio) {
    int idx = (int)(pos_ratio * GRAD_STEPS);
    if (idx < 0) idx = 0;
    if (idx >= GRAD_STEPS) idx = GRAD_STEPS - 1;
    return idx;
}

int get_bottom_gradient_color_idx(float pos_ratio) {
    int idx = (int)(pos_ratio * GRAD_STEPS);
    if (idx < 0) idx = 0;
    if (idx >= GRAD_STEPS) idx = GRAD_STEPS - 1;
    return GRAD_STEPS + idx;
}

int main(int argc, char *argv[]) {
    setlocale(LC_ALL, "");

    pw_init(&argc, &argv);

    struct pw_app app;
    memset(&app, 0, sizeof(app));
    ring_buffer_init(&app.ring_buf);

    app.thread_loop = pw_thread_loop_new("PipeWire Visualizer", NULL);
    if (!app.thread_loop) return 1;

    struct pw_loop *loop = pw_thread_loop_get_loop(app.thread_loop);
    app.context = pw_context_new(loop, NULL, 0);
    if (!app.context) return 1;

    app.core = pw_context_connect(app.context, NULL, 0);
    if (!app.core) return 1;

    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Audio",
        PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_ROLE, "Music",
        "stream.capture.sink", "true",
        PW_KEY_TARGET_OBJECT, "@DEFAULT_AUDIO_SINK@",
        NULL
    );

    app.stream = pw_stream_new(app.core, "Stereo Audio Visualizer", props);
    if (!app.stream) return 1;

    static const struct pw_stream_events stream_events = {
        PW_VERSION_STREAM_EVENTS,
        .process = on_process,
    };

    pw_stream_add_listener(app.stream, &app.stream_listener, &stream_events, &app);

    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    struct spa_audio_info_raw info = SPA_AUDIO_INFO_RAW_INIT(
        .format = SPA_AUDIO_FORMAT_S16_LE,
        .rate = SAMPLE_RATE,
        .channels = 2
    );

    const struct spa_pod *params[1];
    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

    if (pw_stream_connect(app.stream,
                          PW_DIRECTION_INPUT,
                          PW_ID_ANY,
                          PW_STREAM_FLAG_AUTOCONNECT |
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS,
                          params, 1) < 0) {
        return 1;
    }

    pw_thread_loop_start(app.thread_loop);

    double *fft_in_L = fftw_malloc(sizeof(double) * BUFFER_SIZE);
    fftw_complex *fft_out_L = fftw_malloc(sizeof(fftw_complex) * (BUFFER_SIZE / 2 + 1));
    fftw_plan plan_L = fftw_plan_dft_r2c_1d(BUFFER_SIZE, fft_in_L, fft_out_L, FFTW_ESTIMATE);

    double *fft_in_R = fftw_malloc(sizeof(double) * BUFFER_SIZE);
    fftw_complex *fft_out_R = fftw_malloc(sizeof(fftw_complex) * (BUFFER_SIZE / 2 + 1));
    fftw_plan plan_R = fftw_plan_dft_r2c_1d(BUFFER_SIZE, fft_in_R, fft_out_R, FFTW_ESTIMATE);

    int16_t pcm_buffer[BUFFER_SIZE * 2];

    initscr();
    cbreak();
    noecho();
    curs_set(0);
    timeout(0);

    if (!has_colors()) {
        endwin();
        fprintf(stderr, "Error: Terminal does not support color output.\n");
        return 1;
    }

    init_gradient_colors();

    pthread_t np_thread;
    pthread_create(&np_thread, NULL, now_playing_worker, NULL);

    int current_num_bars = 0;
    int (*band_ranges)[2] = NULL;
    float *band_boost = NULL;
    float *heights = NULL;
    float *peaks = NULL;

    int ch;

    while ((ch = getch()) != 'q' && ch != 'Q') {
        if (ring_buffer_read(&app.ring_buf, pcm_buffer, BUFFER_SIZE * 2) < 0) {
            break;
        }

        for (int i = 0; i < BUFFER_SIZE; i++) {
            double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (BUFFER_SIZE - 1)));
            fft_in_L[i] = (pcm_buffer[2 * i]     / 32768.0) * window;
            fft_in_R[i] = (pcm_buffer[2 * i + 1] / 32768.0) * window;
        }

        fftw_execute(plan_L);
        fftw_execute(plan_R);

        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        if (cols < 4 || rows < 4) continue;

        int num_bars = (cols + BAR_GAP) / (BAR_WIDTH + BAR_GAP);
        if (num_bars < 2) num_bars = 2;

        if (num_bars != current_num_bars) {
            current_num_bars = num_bars;

            band_ranges = realloc(band_ranges, sizeof(int[2]) * current_num_bars);
            band_boost  = realloc(band_boost, sizeof(float) * current_num_bars);
            heights     = realloc(heights, sizeof(float) * current_num_bars);
            peaks       = realloc(peaks, sizeof(float) * current_num_bars);

            memset(heights, 0, sizeof(float) * current_num_bars);
            memset(peaks, 0, sizeof(float) * current_num_bars);

            init_bands(current_num_bars, band_ranges, band_boost);
        }

        // Reserve 1 bottom row for MPD track name display
        int visual_rows = rows - 1;
        if (visual_rows % 2 != 0) visual_rows--;
        if (visual_rows < 2) visual_rows = 2;

        int canvas_height = visual_rows * 2;   // Pixel resolution (2 per char row)
        int center_pixel  = canvas_height / 2; // Center axis

        int total_width = (current_num_bars * BAR_WIDTH) + ((current_num_bars - 1) * BAR_GAP);
        int start_x = (cols - total_width) / 2;
        if (start_x < 0) start_x = 0;

        int *canvas = malloc(sizeof(int) * canvas_height * cols);
        if (!canvas) continue;

        for (int k = 0; k < canvas_height * cols; k++) {
            canvas[k] = -1;
        }

        int half_bars = current_num_bars / 2;

        for (int i = 0; i < current_num_bars; i++) {
            fftw_complex *fft_out = (i < half_bars) ? fft_out_L : fft_out_R;

            double band_sum = 0;
            int min_bin = band_ranges[i][0];
            int max_bin = band_ranges[i][1];
            int bin_count = max_bin - min_bin + 1;

            for (int b = min_bin; b <= max_bin; b++) {
                double real = fft_out[b][0];
                double imag = fft_out[b][1];
                double mag = sqrt(real * real + imag * imag);
                band_sum += mag;
            }

            double avg_mag = (band_sum / bin_count) * band_boost[i];

            float compressed_mag = log10f(1.0f + (float)avg_mag);
            float target = (float)(compressed_mag * center_pixel * 0.85f * BAR_LENGTH_SCALE);
            if (target > center_pixel) target = (float)center_pixel;

            if (target > heights[i]) {
                heights[i] += (target - heights[i]) * 0.6f;
            } else {
                heights[i] -= 0.5f;
                if (heights[i] < 0) heights[i] = 0;
            }

            if (heights[i] > peaks[i]) {
                peaks[i] = heights[i];
            } else {
                peaks[i] -= 0.2f;
                if (peaks[i] < 0) peaks[i] = 0;
            }

            int current_hb = (int)heights[i];
            int peak_hb = (int)peaks[i];
            int x_pos = start_x + i * (BAR_WIDTH + BAR_GAP);

            // 1. Top Spectrum (with bounds protection)
            for (int h = 0; h < current_hb; h++) {
                if (h % 2 == 1) continue;

                int p = center_pixel - 1 - h;
                if (p < 0 || p >= canvas_height) continue;

                float ratio = (float)h / (float)center_pixel;
                int col_idx = get_gradient_color_idx(ratio);
                for (int bw = 0; bw < BAR_WIDTH; bw++) {
                    if (x_pos + bw < cols) {
                        canvas[p * cols + (x_pos + bw)] = col_idx;
                    }
                }
            }

            // 2. Bottom Spectrum (with bounds protection)
            for (int h = 0; h < current_hb; h++) {
                if (h % 2 == 1) continue;

                int p = center_pixel + h;
                if (p < 0 || p >= canvas_height) continue;

                float ratio = (float)h / (float)center_pixel;
                int col_idx = get_bottom_gradient_color_idx(ratio);
                for (int bw = 0; bw < BAR_WIDTH; bw++) {
                    if (x_pos + bw < cols) {
                        canvas[p * cols + (x_pos + bw)] = col_idx;
                    }
                }
            }

            // 3. Top Peak Indicator (strictly clamped to canvas boundaries)
            if (peak_hb > 0 && peak_hb <= center_pixel) {
                int aligned_peak = peak_hb - (peak_hb % 2);
                int p_peak = center_pixel - 1 - aligned_peak;
                if (p_peak >= 0 && p_peak < canvas_height) {
                    float ratio = (float)aligned_peak / (float)center_pixel;
                    int col_idx = get_gradient_color_idx(ratio);
                    for (int bw = 0; bw < BAR_WIDTH; bw++) {
                        if (x_pos + bw < cols) {
                            canvas[p_peak * cols + (x_pos + bw)] = col_idx;
                        }
                    }
                }
            }

            // 4. Bottom Peak Indicator (strictly clamped to canvas boundaries)
            if (peak_hb > 0 && peak_hb <= center_pixel) {
                int aligned_peak = peak_hb - (peak_hb % 2);
                int p_peak = center_pixel + aligned_peak;
                if (p_peak >= 0 && p_peak < canvas_height) {
                    float ratio = (float)aligned_peak / (float)center_pixel;
                    int col_idx = get_bottom_gradient_color_idx(ratio);
                    for (int bw = 0; bw < BAR_WIDTH; bw++) {
                        if (x_pos + bw < cols) {
                            canvas[p_peak * cols + (x_pos + bw)] = col_idx;
                        }
                    }
                }
            }
        }

        erase();

        // Render virtual canvas to ncurses screen
        for (int r = 0; r < visual_rows; r++) {
            for (int c = 0; c < cols; c++) {
                int c_top = canvas[(2 * r) * cols + c];
                int c_bot = canvas[(2 * r + 1) * cols + c];

                if (c_top == -1 && c_bot == -1) continue;

                if (c_top != -1 && c_bot == -1) {
                    int pair_id = get_pair(c_top, -1);
                    attron(COLOR_PAIR(pair_id) | A_BOLD);
                    mvaddwstr(r, c, L"▀");
                    attroff(COLOR_PAIR(pair_id) | A_BOLD);
                } else if (c_top == -1 && c_bot != -1) {
                    int pair_id = get_pair(c_bot, -1);
                    attron(COLOR_PAIR(pair_id) | A_BOLD);
                    mvaddwstr(r, c, L"▄");
                    attroff(COLOR_PAIR(pair_id) | A_BOLD);
                } else {
                    if (c_top == c_bot) {
                        int pair_id = get_pair(c_top, -1);
                        attron(COLOR_PAIR(pair_id) | A_BOLD);
                        mvaddwstr(r, c, L"█");
                        attroff(COLOR_PAIR(pair_id) | A_BOLD);
                    } else {
                        int pair_id = get_pair(c_top, c_bot);
                        attron(COLOR_PAIR(pair_id) | A_BOLD);
                        mvaddwstr(r, c, L"▀");
                        attroff(COLOR_PAIR(pair_id) | A_BOLD);
                    }
                }
            }
        }

        free(canvas);

        // Render MPD Song Track Name at bottom
        char display_np[256] = {0};
        pthread_mutex_lock(&np_mutex);
        strncpy(display_np, now_playing, sizeof(display_np) - 1);
        pthread_mutex_unlock(&np_mutex);

        if (display_np[0] != '\0') {
            wchar_t wbuf[256] = {0};
            if (mbstowcs(wbuf, display_np, 255) != (size_t)-1) {
                int text_width = wcswidth(wbuf, 255);
                if (text_width < 0) text_width = (int)strlen(display_np);

                if (text_width > cols) {
                    wbuf[cols] = L'\0';
                    text_width = cols;
                }

                int text_x = (cols - text_width) / 2;
                if (text_x < 0) text_x = 0;

                move(rows - 1, 0);
                clrtoeol();

                attron(A_BOLD);
                mvaddwstr(rows - 1, text_x, wbuf);
                attroff(A_BOLD);
            }
        }

        refresh();
    }

    keep_running = 0;
    pthread_cond_broadcast(&app.ring_buf.cond);

    pthread_join(np_thread, NULL);

    endwin();

    pw_thread_loop_stop(app.thread_loop);
    pw_stream_destroy(app.stream);
    pw_core_disconnect(app.core);
    pw_context_destroy(app.context);
    pw_thread_loop_destroy(app.thread_loop);
    pw_deinit();

    ring_buffer_free(&app.ring_buf);

    free(band_ranges);
    free(band_boost);
    free(heights);
    free(peaks);

    fftw_destroy_plan(plan_L);
    fftw_free(fft_in_L);
    fftw_free(fft_out_L);

    fftw_destroy_plan(plan_R);
    fftw_free(fft_in_R);
    fftw_free(fft_out_R);

    return 0;
}