int main(void)
{
    struct trace_collector_bpf *skel = trace_collector_bpf__open_and_load();
    trace_collector_bpf__attach(skel);
    struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.events),
                                               (ring_buffer_sample_fn)on_sample, NULL, NULL);
    while (!exiting) ring_buffer__poll(rb, 100 /* ms */);
    return 0;
}

static int on_sample(void *ctx, void *data, size_t len)
{
    handle_event((struct trace_event *)data);
    return 0;
}
