#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include <ncurses.h>
#include <pulse/simple.h>
#include <pulse/error.h>
#include <fftw3.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_BARS 30         // Change to any integer (e.g. 10, 16, 24, 32, 40)
#define BAR_GAP 1            // Space gap between columns
#define BAR_WIDTH 2          // Number of half-block characters per bar
#define BAR_LENGTH_SCALE 1.0f // Multiplier to adjust overall bar height/length
#define SAMPLE_RATE 44100
#define BUFFER_SIZE 2048     // ~21.53Hz per FFT bin

#define GRAD_STEPS 32

// Dynamic frequency range & boost tables
int band_ranges[NUM_BARS][2];
float band_boost[NUM_BARS];

// Dynamically generate logarithmic frequency bands mirrored horizontally (Low outer, High center)
void init_bands(int num_bars) {
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

        // Outer to Center (Left side)
        band_ranges[i][0] = b_start;
        band_ranges[i][1] = b_end;

        // Center to Outer (Right side mirrored)
        int mirror_idx = num_bars - 1 - i;
        band_ranges[mirror_idx][0] = b_start;
        band_ranges[mirror_idx][1] = b_end;

        // Dynamic boost scaling from 1.0 (bass) to 3.5 (treble)
        float boost_val = 1.0f + 2.5f * ((float)i / (float)(half > 1 ? half - 1 : 1));
        band_boost[i] = boost_val;
        band_boost[mirror_idx] = boost_val;
    }

    // Odd bar fallback for middle bar
    if (num_bars % 2 != 0) {
        band_ranges[half][0] = max_bin / 2;
        band_ranges[half][1] = max_bin;
        band_boost[half] = 3.5f;
    }
}

// Threading & Now Playing state
char now_playing[256] = "";
pthread_mutex_t np_mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int keep_running = 1;

void* now_playing_worker(void* arg) {
    const char* cmd = "mpc -p 8600 current 2>/dev/null";

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
        pthread_mutex_unlock(&np_mutex);

        for (int i = 0; i < 5 && keep_running; i++) {
            usleep(100000);
        }
    }
    return NULL;
}

void init_gradient_colors() {
    start_color();
    use_default_colors();

    static const float palette[5][3] = {
        {553.0f,  894.0f, 110.0f}, // YellowGreen -> #8de41c
        {1000.0f, 1000.0f,   0.0f}, // Yellow      -> #ffff00
        {1000.0f,  647.0f,   0.0f}, // Orange      -> #ffa500
        {1000.0f,  271.0f,   0.0f}, // Orange-Red  -> #ff4500
        {1000.0f,    0.0f,   0.0f}  // Bright Red  -> #ff0000
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

            short color_id = 16 + pos;
            short pair_id  = 1 + pos;

            init_color(color_id, r, g, b);
            init_pair(pair_id, color_id, -1);
        }
    } else if (has_colors() && COLORS >= 256) {
        for (int pos = 0; pos < GRAD_STEPS; pos++) {
            short pair_id = 1 + pos;
            float t = (float)pos / (float)(GRAD_STEPS - 1);

            int color_code;
            if (t < 0.20f)      color_code = 118; // YellowGreen
            else if (t < 0.45f) color_code = 226; // Yellow
            else if (t < 0.70f) color_code = 208; // Orange
            else if (t < 0.88f) color_code = 202; // Orange-Red
            else                color_code = 196; // Bright Red

            init_pair(pair_id, color_code, -1);
        }
    } else {
        for (int i = 1; i <= GRAD_STEPS; i++) {
            init_pair(i, COLOR_RED, -1);
        }
    }
}

int get_gradient_color(float pos_ratio) {
    int pos_idx = (int)(pos_ratio * GRAD_STEPS);
    if (pos_idx < 0) pos_idx = 0;
    if (pos_idx >= GRAD_STEPS) pos_idx = GRAD_STEPS - 1;

    return 1 + pos_idx;
}

int main() {
    setlocale(LC_ALL, "");
    init_bands(NUM_BARS);

    // 2 Channels for Stereo input
    static const pa_sample_spec ss = {
        .format = PA_SAMPLE_S16LE,
        .rate = SAMPLE_RATE,
        .channels = 2
    };

    int pa_error;
    pa_simple *pa_stream = pa_simple_new(
        NULL, "Stereo Audio Visualizer", PA_STREAM_RECORD, 
        NULL, "Spectrum", &ss, NULL, NULL, &pa_error
    );

    if (!pa_stream) {
        fprintf(stderr, "PulseAudio Error: %s\n", pa_strerror(pa_error));
        return 1;
    }

    // FFT setups for Left & Right channels
    double *fft_in_L = fftw_malloc(sizeof(double) * BUFFER_SIZE);
    fftw_complex *fft_out_L = fftw_malloc(sizeof(fftw_complex) * (BUFFER_SIZE / 2 + 1));
    fftw_plan plan_L = fftw_plan_dft_r2c_1d(BUFFER_SIZE, fft_in_L, fft_out_L, FFTW_ESTIMATE);

    double *fft_in_R = fftw_malloc(sizeof(double) * BUFFER_SIZE);
    fftw_complex *fft_out_R = fftw_malloc(sizeof(fftw_complex) * (BUFFER_SIZE / 2 + 1));
    fftw_plan plan_R = fftw_plan_dft_r2c_1d(BUFFER_SIZE, fft_in_R, fft_out_R, FFTW_ESTIMATE);

    // Interleaved stereo buffer (2 * BUFFER_SIZE samples)
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

    float heights[NUM_BARS];
    float peaks[NUM_BARS];
    memset(heights, 0, sizeof(heights));
    memset(peaks, 0, sizeof(peaks));

    int ch;

    while ((ch = getch()) != 'q' && ch != 'Q') {
        if (pa_simple_read(pa_stream, pcm_buffer, sizeof(pcm_buffer), &pa_error) < 0) {
            break;
        }

        // De-interleave PCM stereo audio and apply Hanning window
        for (int i = 0; i < BUFFER_SIZE; i++) {
            double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (BUFFER_SIZE - 1)));
            fft_in_L[i] = (pcm_buffer[2 * i]     / 32768.0) * window; // Left channel
            fft_in_R[i] = (pcm_buffer[2 * i + 1] / 32768.0) * window; // Right channel
        }

        fftw_execute(plan_L);
        fftw_execute(plan_R);

        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        int visual_area = rows - 2;
        int center_y = visual_area / 2;
        int max_height = center_y; 
        if (max_height < 2) max_height = 2;

        int total_width = (NUM_BARS * BAR_WIDTH) + ((NUM_BARS - 1) * BAR_GAP);
        int start_x = (cols - total_width) / 2;
        if (start_x < 0) start_x = 0;

        erase();

        int half_bars = NUM_BARS / 2;

        for (int i = 0; i < NUM_BARS; i++) {
            // Assign Left FFT to left side bars, Right FFT to right side bars
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
            float target = (float)(compressed_mag * max_height * 0.85f * BAR_LENGTH_SCALE);
            if (target > max_height) target = (float)max_height;

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

            int current_height = (int)heights[i];
            int x_pos = start_x + i * (BAR_WIDTH + BAR_GAP);

            for (int y = 0; y < current_height; y++) {
                float pos_ratio = (float)y / (float)max_height;
                int color = get_gradient_color(pos_ratio);

                attron(COLOR_PAIR(color) | A_BOLD);
                for (int bw = 0; bw < BAR_WIDTH; bw++) {
                    if (center_y - y >= 0) {
                        mvprintw(center_y - y, x_pos + bw, "▀");
                    }
                    if (center_y + y < rows - 1) {
                        mvprintw(center_y + y, x_pos + bw, "▀");
                    }
                }
                attroff(COLOR_PAIR(color) | A_BOLD);
            }

            int peak_y = (int)peaks[i];
            if (peak_y > 0 && peak_y < max_height) {
                float peak_pos_ratio = (float)peak_y / (float)max_height;
                int peak_color = get_gradient_color(peak_pos_ratio);

                attron(COLOR_PAIR(peak_color) | A_BOLD);
                for (int bw = 0; bw < BAR_WIDTH; bw++) {
                    if (center_y - peak_y >= 0) {
                        mvprintw(center_y - peak_y, x_pos + bw, "▀");
                    }
                    if (center_y + peak_y < rows - 1) {
                        mvprintw(center_y + peak_y, x_pos + bw, "▀");
                    }
                }
                attroff(COLOR_PAIR(peak_color) | A_BOLD);
            }
        }

        char display_np[256] = {0};
        pthread_mutex_lock(&np_mutex);
        strncpy(display_np, now_playing, sizeof(display_np) - 1);
        pthread_mutex_unlock(&np_mutex);

        if (display_np[0] != '\0') {
            int np_len = (int)strlen(display_np);
            if (np_len > cols) {
                display_np[cols] = '\0';
                np_len = cols;
            }
            int text_x = (cols - np_len) / 2;
            if (text_x < 0) text_x = 0;

            attron(A_BOLD);
            mvprintw(rows - 1, text_x, "%s", display_np);
            attroff(A_BOLD);
        }

        refresh();
    }

    keep_running = 0;
    pthread_join(np_thread, NULL);

    endwin();

    fftw_destroy_plan(plan_L);
    fftw_free(fft_in_L);
    fftw_free(fft_out_L);

    fftw_destroy_plan(plan_R);
    fftw_free(fft_in_R);
    fftw_free(fft_out_R);

    pa_simple_free(pa_stream);

    return 0;
}

