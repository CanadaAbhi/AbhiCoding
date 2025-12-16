#define LANES 4
#define COLS  60
enum { LANE_CPU, LANE_IRQ, LANE_GPU, LANE_DMA };

static void mark(char grid[LANES][COLS + 1], int lane, u64 t0, u64 t1, u64 win_start, u64 win_ns)
{
    if (!t0 || !t1 || t1 <= t0) return;
    int c0 = (int)(((__int128)(t0 - win_start) * COLS) / win_ns);
    int c1 = (int)(((__int128)(t1 - win_start) * COLS) / win_ns);
    c0 = c0 < 0 ? 0 : (c0 >= COLS ? COLS - 1 : c0);
    c1 = c1 < 0 ? 0 : (c1 >= COLS ? COLS - 1 : c1);
    for (int c = c0; c <= c1; c++) grid[lane][c] = '#';
}

static void render_timeline(struct job_trace *j)
{
    char grid[LANES][COLS + 1];
    for (int l = 0; l < LANES; l++) { memset(grid[l], '-', COLS); grid[l][COLS] = '\0'; }

    u64 win_start = j->t_syscall_enter;
    u64 win_ns    = j->t_syscall_exit - j->t_syscall_enter;

    /* CPU lane here specifically means "the submitting/waiting thread is
     * on-CPU," reconstructed from sched_switch, not a full system-wide
     * per-core occupancy view -- that's a straightforward extension
     * (key the grid by cpu id across ALL threads instead of by job) but
     * out of scope for a single-job trace rendering. */
    mark(grid, LANE_CPU, j->t_syscall_enter, j->t_submit_enter, win_start, win_ns);
    mark(grid, LANE_CPU, j->t_sched_in,      j->t_syscall_exit, win_start, win_ns);
    mark(grid, LANE_IRQ, j->t_irq_entry,     j->t_softirq_entry, win_start, win_ns);
    mark(grid, LANE_GPU, j->t_gpu_start,     j->t_gpu_complete,  win_start, win_ns);
    mark(grid, LANE_DMA, j->t_dma_start,     j->t_dma_complete,  win_start, win_ns);

    printf("CPU %s\n", grid[LANE_CPU]);
    printf("IRQ %s\n", grid[LANE_IRQ]);
    printf("GPU %s\n", grid[LANE_GPU]);
    printf("DMA %s\n", grid[LANE_DMA]);
}
