# Next-matchup outlook repair — October 24, 2026

The VJ Edgecombe Points Over 13.5 job for Milwaukee at Philadelphia failed in the hosted worker. Reproduction on the deployed Linux process raised `RuntimeError: can't start new thread` in OpponentHistory's six-thread pool. The worker retains a 512 MiB address-space limit; glibc allocator arena reservations exhausted that virtual address space intermittently, even with cached data.

The child-process environment now sets `MALLOC_ARENA_MAX=2` before Python starts. CPU, address-space, file-size and request-time limits remain unchanged; credentials remain excluded. Three consecutive bounded Linux replays passed in 2.8 seconds each after the change. Only this environment change was applied to production, with a hash-checked backup and healthy restart; unrelated deployed/local differences were preserved.

The actual iOS retry completed in 2.98 seconds, persisted the snapshot, and displayed recent production, opponent history, minutes, opponent scoring and the recorded DraftKings +101 quote. This selection appropriately has no numerical estimate: interval coverage, Brier comparison and calibration did not pass. The fix does not lower validation gates or invent probabilities. The refreshed simulator retained all four Builder picks.

The app now describes worker failure as a service interruption with a retry on the same selection, and distinguishes Try again from Resume outlook. Fifty targeted Python service/queue/context tests passed. Full Swift release tests and release archive results are recorded separately.
