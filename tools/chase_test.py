"""Active localization chase test — 5 rounds of random positions."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml, tempfile, os, sys, time
sys.path.insert(0, 'D:/shengxuedingwei2')
from src.pipeline.simulator import BinauralSimulator
from src.localization.active_locator import ActiveLocator

N_ROUNDS = 5
N_PER_ROUND = 30  # 150 total
CHASE_THRESH = 10.0  # deg — DOA within this = locked

np.random.seed(42)
all_results = []

for round_idx in range(N_ROUNDS):
    print(f'\n=== Round {round_idx+1}/{N_ROUNDS} ===')
    round_results = []
    for i in range(N_PER_ROUND):
        az = np.random.uniform(-180, 180)
        dist = np.random.uniform(1.5, 3.0)
        rad = np.deg2rad(az)
        px = float(4 + dist * np.sin(rad))
        py = float(4 + dist * np.cos(rad))

        cfg = {
            'name': f't{round_idx}_{i}', 'description': '',
            'room': {'dimensions': [8, 8, 3], 'absorption': 1.0, 'max_order': 0},
            'microphone': {'head_center': [4, 4, 1.5], 'head_radius': 0.09, 'hrtf_mode': 'analytical'},
            'source': {'generator': 'human_voice', 'duration': 2.0, 'sample_rate': 44100,
                       'F0': 150, 'formants': [850, 1700, 2600, 3600]},
            'motion': {'enabled': False, 'type': 'static', 'position': [px, py, 1.5]},
            'output': {'sample_rate': 44100, 'visualize': False},
            'noise': {'enabled': False},
            'localization': {'active_head': True, 'frame_duration_ms': 50.0, 'frame_hop_ms': 25.0},
            'evaluation': {'enabled': False},
        }

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(cfg, tmp); tmp.close()

        try:
            sim = BinauralSimulator(tmp.name)
            stereo, _, _, _, _ = sim.run()
            loc = ActiveLocator(44100)
            result = loc.localize(stereo)
            doa = result.doa_estimated

            # After detection frame (31 for back, 12 for front): check if DOA converges
            detect = 31 if abs(az) > 90 else 12
            doa_after = doa[detect:] if detect < len(doa) else doa

            # Compute lock frame (first frame where |DOA| < CHASE_THRESH)
            lock_frame = None
            for j, d in enumerate(doa_after):
                if abs(d) < CHASE_THRESH:
                    lock_frame = detect + j
                    break

            chase_time_s = (lock_frame - detect) * 0.025 if lock_frame else None
            locked = lock_frame is not None
            flash_used = abs(az) > 90

            round_results.append({
                'az': az, 'dist': dist, 'locked': locked,
                'chase_time': chase_time_s, 'flash': flash_used,
                'doa_final': float(np.mean(doa_after[-10:])) if len(doa_after) >= 10 else float(np.mean(doa_after)),
            })
        except Exception as e:
            round_results.append({'az': az, 'dist': dist, 'locked': False, 'error': str(e)})
        finally:
            os.unlink(tmp.name)

    locked_n = sum(1 for r in round_results if r.get('locked'))
    chase_times = [r['chase_time'] for r in round_results if r.get('locked') and r['chase_time'] is not None]
    flash_n = sum(1 for r in round_results if r.get('flash'))
    flash_locked = sum(1 for r in round_results if r.get('flash') and r.get('locked'))

    print(f'  Locked: {locked_n}/{N_PER_ROUND}')
    print(f'  Avg chase time: {np.mean(chase_times):.2f}s' if chase_times else '  No locks')
    print(f'  Flash scenarios: {flash_n}, locked: {flash_locked}')

    all_results.extend(round_results)

# Summary
total = len(all_results)
locked_total = sum(1 for r in all_results if r.get('locked'))
flash_total = sum(1 for r in all_results if r.get('flash'))
flash_ok = sum(1 for r in all_results if r.get('flash') and r.get('locked'))
chase_t_all = [r['chase_time'] for r in all_results if r.get('locked') and r['chase_time'] is not None]

print(f'\n===== FINAL REPORT =====')
print(f'Total tests: {total}')
print(f'Overall lock rate: {locked_total}/{total} ({locked_total/total*100:.1f}%)')
print(f'Avg chase time: {np.mean(chase_t_all):.2f}s' if chase_t_all else 'N/A')
print(f'Back-hemisphere (flash): {flash_total}, locked: {flash_ok} ({flash_ok/flash_total*100:.1f}%)' if flash_total else 'N/A')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for r in all_results:
    c = 'g' if r.get('locked') else 'r'
    ax.scatter(r['az'], r['dist'], c=c, s=20, alpha=0.6, edgecolors='none')
ax.axvline(x=0, c='w', ls='--', lw=1)
ax.axvline(x=90, c='y', ls=':', lw=1)
ax.axvline(x=-90, c='y', ls=':', lw=1)
ax.set_xlabel('True Azimuth (deg)'); ax.set_ylabel('Distance (m)')
ax.set_title(f'Chase Lock: {locked_total}/{total} ({locked_total/total*100:.0f}%)')
ax.set_xlim(-185, 185); ax.grid(True, alpha=0.3)

ax2 = axes[1]
bins = np.arange(-180, 181, 30)
for lo in np.arange(-180, 180, 30):
    hi = lo + 30
    br = [r for r in all_results if lo <= r['az'] < hi]
    if br:
        acc = sum(1 for r in br if r.get('locked')) / len(br) * 100
        ax2.bar((lo+hi)/2, acc, width=25, color='steelblue', edgecolor='white')
ax2.axhline(y=100, c='g', ls='--', lw=1)
ax2.set_xlabel('True Azimuth (deg)'); ax2.set_ylabel('Lock Rate (%)')
ax2.set_title('Lock Rate by Angle'); ax2.set_ylim(0, 110); ax2.grid(True, alpha=0.3)

plt.suptitle(f'Active Localization Chase Test — {total} positions', fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'D:/shengxuedingwei2/results/chase_test.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
