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

#define NUM_BARS 20
#define BAR_GAP 1        // 1 space gap between columns
#define SAMPLE_RATE 44100
#define BUFFER_SIZE 2048 // ~21.53Hz per FFT bin

#define GRAD_STEPS 32

// 20 columns spanning 30Hz to 15kHz horizontally mirrored (Low outer, High center)
const int BAND_RANGES[NUM_BARS][2] = {
    // Cols 1-10 (Low -> High)
    {1, 2},     // ~21 Hz   - 55 Hz
    {3, 4},     // ~55 Hz   - 104 Hz
    {5, 8},     // ~104 Hz  - 193 Hz
    {9, 16},    // ~193 Hz  - 359 Hz
    {17, 30},   // ~359 Hz  - 671 Hz
    {31, 57},   // ~671 Hz  - 1251 Hz
    {58, 107},  // ~1251 Hz - 2331 Hz
    {108, 201}, // ~2331 Hz - 4343 Hz
    {202, 375}, // ~4343 Hz - 8091 Hz
    {376, 697}, // ~8091 Hz - 15000 Hz
    // Cols 11-20 (High -> Low)
    {376, 697}, // ~8091 Hz - 15000 Hz
    {202, 375}, // ~4343 Hz - 8091 Hz
    {108, 201}, // ~2331 Hz - 4343 Hz
    {58, 107},  // ~1251 Hz - 2331 Hz
    {31, 57},   // ~671 Hz  - 1251 Hz
    {17, 30},   // ~359 Hz  - 671 Hz
    {9, 16},    // ~193 Hz  - 359 Hz
    {5, 8},     // ~104 Hz  - 193 Hz
    {3, 4},     // ~55 Hz   - 104 Hz
    {1, 2}      // ~21 Hz   - 55 Hz
};

const float BAND_BOOST[NUM_BARS] = {
    1.0f, 1.0f, 1.0f, 1.1f, 1.3f, 1.5f, 1.8f, 2.2f, 2.8f, 3.5f,
    3.5f, 2.8f, 2.2f, 1.8f, 1.5f, 1.3f, 1.1f, 1.0f, 1.0f, 1.0f
};

// Threading & Now Playing state
char now_playing[256] = "";
pthread_mutex_t np_mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int keep_running = 1;

void* now_playing_worker(void* arg) {
    // Direct mpc query targeting port 8600
    const char* cmd = "mpc -p 8600 current 2>/dev/null";

    while (keep_running) {
        FILE *fp = popen(cmd, "r");
        char buf[256] = {0};
        if (fp) {
            if (fgets(buf, sizeof(buf) - 1, fp) != NULL) {
                buf[strcspn(buf, "\r\n")] = 0; // Strip trailing newline
            }
            pclose(fp);
        }

        pthread_mutex_lock(&np_mutex);
        strncpy(now_playing, buf, sizeof(now_playing) - 1);
        pthread_mutex_unlock(&np_mutex);

        // Poll every 500ms
        for (int i = 0; i < 5 && keep_running; i++) {
            usleep(100000); // 100ms chunks
        }
    }
    return NULL;
}

void init_gradient_colors() {
    start_color();
    use_default_colors();

    // Reversed Palette Keyframes mapped to ncurses 0..1000 scale:
    // 0: #8de41c -> (553, 894, 110)
    // 1: #f6287d -> (965, 157, 490)
    // 2: #b915cc -> (725,  82, 800)
    // 3: #210456 -> (129,  16, 337)
    // 4: #0a0324 -> ( 39,  12, 141)
    static const float palette[5][3] = {
        {553.0f, 894.0f, 110.0f},
        {965.0f, 157.0f, 490.0f},
        {725.0f,  82.0f, 800.0f},
        {129.0f,  16.0f, 337.0f},
        { 39.0f,  12.0f, 141.0f}
    };

    if (has_colors() && can_change_color()) {
        for (int pos = 0; pos < GRAD_STEPS; pos++) {
            float t = (float)pos / (float)(GRAD_STEPS - 1);
            
            // Map t [0..1] across 4 segments
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
            if (t < 0.10f)      color_code = 118; // Neon green
            else if (t < 0.25f) color_code = 198; // Neon pink
            else if (t < 0.50f) color_code = 128; // Bright magenta
            else if (t < 0.75f) color_code = 54;  // Deep violet
            else                color_code = 234; // Deep purple-black

            init_pair(pair_id, color_code, -1);
        }
    } else {
        for (int i = 1; i <= GRAD_STEPS; i++) {
            init_pair(i, COLOR_MAGENTA, -1);
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

    static const pa_sample_spec ss = {
        .format = PA_SAMPLE_S16LE,
        .rate = SAMPLE_RATE,
        .channels = 1
    };

    int pa_error;
    pa_simple *pa_stream = pa_simple_new(
        NULL, "20-Bar Visualizer", PA_STREAM_RECORD, 
        NULL, "Spectrum", &ss, NULL, NULL, &pa_error
    );

    if (!pa_stream) {
        fprintf(stderr, "PulseAudio Error: %s\n", pa_strerror(pa_error));
        return 1;
    }

    double *fft_in = fftw_malloc(sizeof(double) * BUFFER_SIZE);
    fftw_complex *fft_out = fftw_malloc(sizeof(fftw_complex) * (BUFFER_SIZE / 2 + 1));
    fftw_plan plan = fftw_plan_dft_r2c_1d(BUFFER_SIZE, fft_in, fft_out, FFTW_ESTIMATE);
    int16_t pcm_buffer[BUFFER_SIZE];

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

    // Start background MPD listener thread
    pthread_t np_thread;
    pthread_create(&np_thread, NULL, now_playing_worker, NULL);

    float heights[NUM_BARS] = {0};
    float peaks[NUM_BARS] = {0};
    int ch;

    while ((ch = getch()) != 'q' && ch != 'Q') {
        if (pa_simple_read(pa_stream, pcm_buffer, sizeof(pcm_buffer), &pa_error) < 0) {
            break;
        }

        for (int i = 0; i < BUFFER_SIZE; i++) {
            double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (BUFFER_SIZE - 1)));
            fft_in[i] = (pcm_buffer[i] / 32768.0) * window;
        }

        fftw_execute(plan);

        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        // Reserve row (rows - 1) for track info at bottom
        int visual_area = rows - 2;
        int center_y = visual_area / 2;
        int max_height = center_y; 
        if (max_height < 2) max_height = 2;

        int bar_width = (cols - (NUM_BARS - 1) * BAR_GAP) / NUM_BARS;
        if (bar_width < 1) bar_width = 1;

        int total_width = (NUM_BARS * bar_width) + ((NUM_BARS - 1) * BAR_GAP);
        int start_x = (cols - total_width) / 2;
        if (start_x < 0) start_x = 0;

        erase();

        for (int i = 0; i < NUM_BARS; i++) {
            double band_sum = 0;
            int min_bin = BAND_RANGES[i][0];
            int max_bin = BAND_RANGES[i][1];
            int bin_count = max_bin - min_bin + 1;

            for (int b = min_bin; b <= max_bin; b++) {
                double real = fft_out[b][0];
                double imag = fft_out[b][1];
                double mag = sqrt(real * real + imag * imag);
                band_sum += mag;
            }

            double avg_mag = (band_sum / bin_count) * BAND_BOOST[i];

            float compressed_mag = log10f(1.0f + (float)avg_mag);
            float target = (float)(compressed_mag * max_height * 0.85f);
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
            int x_pos = start_x + i * (bar_width + BAR_GAP);

            // Render bars growing symmetrically UP and DOWN from center_y
            for (int y = 0; y < current_height; y++) {
                float pos_ratio = (float)y / (float)max_height;
                int color = get_gradient_color(pos_ratio);

                attron(COLOR_PAIR(color) | A_BOLD);
                for (int bw = 0; bw < bar_width; bw++) {
                    if (center_y - y >= 0) {
                        mvprintw(center_y - y, x_pos + bw, "▀");
                    }
                    if (center_y + y < rows - 1) {
                        mvprintw(center_y + y, x_pos + bw, "▀");
                    }
                }
                attroff(COLOR_PAIR(color) | A_BOLD);
            }

            // Render mirrored peak indicators
            int peak_y = (int)peaks[i];
            if (peak_y > 0 && peak_y < max_height) {
                float peak_pos_ratio = (float)peak_y / (float)max_height;
                int peak_color = get_gradient_color(peak_pos_ratio);

                attron(COLOR_PAIR(peak_color) | A_BOLD);
                for (int bw = 0; bw < bar_width; bw++) {
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

        // Render MPD track string centered at the bottom
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
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
    pa_simple_free(pa_stream);

    return 0;
}

