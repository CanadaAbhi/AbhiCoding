# Runtime PM: force autosuspend, then wake via access, repeat N times, collect debugfs latency
for i in $(seq 1 100); do
  echo auto > /sys/devices/platform/dev_a/power/control
  sleep 0.3                                   # let autosuspend timer fire
  cat /sys/devices/platform/dev_a/dev_a_ctl > /dev/null   # touch it -> forces resume
done
cat /sys/kernel/debug/pm_lab/stats

# System suspend/resume, waking via Device A's IRQ as wakeup source
rtcwake -m mem -s 5 &          # or: trigger fake wake-IRQ from a debugfs "poke" file
echo mem > /sys/power/state
# on resume, read debugfs latency histogram for the full suspend->resume round trip
