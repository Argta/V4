"""Phase 1+2+3 integrated test with report figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml, tempfile, os, sys, time
sys.path.insert(0, 'D:/shengxuedingwei2')
from src.pipeline.simulator import BinauralSimulator
from src.localization.active_locator import ActiveLocator

N_TESTS = 50

np.random.seed(2024)
results = []
print(f'Running {N_TESTS} tests...')
t0 = time.time()

for i in range(N_TESTS):
    az = np.random.uniform(-180, 180)
    dist = np.random.uniform(1.5, 3.0)
    rad = np.deg2rad(az)
    px = float(4 + dist * np.sin(rad))
    py = float(4 + dist * np.cos(rad))

    cfg = {
        'name': f't{i}', 'description': '',
        'room': {'dimensions': [8, 8, 3], 'absorption': 1.0, 'max_order': 0},
        'microphone': {'head_center': [4, 4, 1.5], 'head_radius': 0.09, 'hrtf_mode': 'analytical'},
        'source': {'generator': 'human_voice', 'duration': 2.5, 'sample_rate': 44100,
                   'F0': 150, 'formants': [850, 1700, 2600, 3600]},
        'motion': {'enabled': False, 'type': 'static', 'position': [px, py, 1.5]},
        'output': {'sample_rate': 44100, 'visualize': False}, 'noise': {'enabled': False},
        'localization': {'active_head': True}, 'evaluation': {'enabled': False},
    }

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp); tmp.close()
    try:
        sim = BinauralSimulator(tmp.name)
        stereo, _, _, _, _ = sim.run()
        loc = ActiveLocator(44100)
        result = loc.localize(stereo)

        locked = result.lock_achieved
        lock_at = result.lock_frame
        doa = result.doa_estimated
        doa_s = result.doa_smoothed if result.doa_smoothed is not None else doa
        chase_s = ((lock_at - 12) * 0.025) if lock_at >= 0 else None
        doa_final = float(np.mean(doa_s[-10:])) if len(doa_s) >= 10 else float(np.mean(doa))

        # Get Phase 1+2 results
        sl = loc._phase1_detect_side(stereo)
        sb = loc._phase2_detect_fb(stereo, sl, start_frame=4)

        results.append({
            'az': az, 'dist': dist, 'locked': locked,
            'chase_time': chase_s, 'doa_final': doa_final,
            'phase1_correct': sl == (az < 0),
            'phase2_correct': sb == (abs(az) > 90),
            'quadrant': ('F' if abs(az)<=90 else 'B')+('R' if az>=0 else 'L'),
        })
    except Exception as e:
        results.append({'az': az, 'dist': dist, 'locked': False, 'error': str(e)})
    finally:
        os.unlink(tmp.name)

    if (i+1) % 10 == 0:
        print(f'  {i+1}/{N_TESTS}')

elapsed = time.time() - t0
locked_n = sum(1 for r in results if r.get('locked'))
chase_t = [r['chase_time'] for r in results if r.get('locked') and r.get('chase_time')]
p1_ok = sum(1 for r in results if r.get('phase1_correct'))
p2_ok = sum(1 for r in results if r.get('phase2_correct'))

print(f'\nResults: {locked_n}/{N_TESTS} locked ({locked_n/N_TESTS*100:.0f}%)')
print(f'Phase1 LR: {p1_ok}/{N_TESTS} ({p1_ok/N_TESTS*100:.0f}%)')
print(f'Phase2 FB: {p2_ok}/{N_TESTS} ({p2_ok/N_TESTS*100:.0f}%)')
print(f'Avg chase: {np.mean(chase_t):.2f}s' if chase_t else 'N/A')
print(f'Time: {elapsed:.0f}s')

# === FIGURE ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Lock scatter
ax = axes[0, 0]
for r in results:
    c = 'g' if r.get('locked') else 'r'
    ax.scatter(r['az'], r['dist'], c=c, s=25, alpha=0.6, edgecolors='none')
ax.axvline(x=0, c='w', ls='--', lw=1)
ax.axvline(x=90, c='y', ls=':', lw=1); ax.axvline(x=-90, c='y', ls=':', lw=1)
ax.set_xlabel('True Azimuth (deg)'); ax.set_ylabel('Distance (m)')
ax.set_title(f'Phase 3 Lock: {locked_n}/{N_TESTS} ({locked_n/N_TESTS*100:.0f}%)')
ax.set_xlim(-185, 185); ax.grid(True, alpha=0.3)

# 2. Lock rate by angle
ax = axes[0, 1]
for lo in np.arange(-180, 180, 30):
    hi = lo + 30; br = [r for r in results if lo <= r['az'] < hi]
    if br:
        acc = sum(1 for r in br if r.get('locked')) / len(br) * 100
        ax.bar((lo+hi)/2, acc, width=25, color='steelblue', edgecolor='white')
ax.axhline(y=100, c='g', ls='--', lw=1); ax.axhline(y=50, c='gray', ls=':', lw=1)
ax.set_xlabel('True Azimuth (deg)'); ax.set_ylabel('Lock Rate (%)')
ax.set_title('Lock Rate by Angle Bin'); ax.set_ylim(0, 110); ax.grid(True, alpha=0.3)

# 3. Chase time histogram
ax = axes[0, 2]
if chase_t:
    ax.hist(chase_t, bins=15, color='steelblue', edgecolor='white')
    ax.axvline(x=np.mean(chase_t), c='r', ls='--', lw=1.5, label=f'Mean={np.mean(chase_t):.2f}s')
    ax.set_xlabel('Chase Time (s)'); ax.set_ylabel('Count')
    ax.set_title('Chase Time Distribution'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
else:
    ax.set_title('No locks recorded')

# 4. Quadrant breakdown
ax = axes[1, 0]
quadrants = {'FR': (0, 90), 'FL': (-90, 0), 'BR': (90, 180), 'BL': (-180, -90)}
colors = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']
for (qname, (lo, hi)), c in zip(quadrants.items(), colors):
    br = [r for r in results if lo <= r['az'] < hi]
    if br:
        acc = sum(1 for r in br if r.get('locked')) / len(br) * 100
        ax.bar(qname, acc, color=c, edgecolor='white', width=0.6)
        ax.text(qname, acc+2, f'{acc:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Lock Rate (%)'); ax.set_title('Lock Rate by Quadrant')
ax.set_ylim(0, 115); ax.grid(True, alpha=0.3, axis='y')

# 5. Phase accuracy
ax = axes[1, 1]
labels = ['Phase1\n(L/R)', 'Phase2\n(F/B)', 'Phase3\n(Lock)']
values = [p1_ok/N_TESTS*100, p2_ok/N_TESTS*100, locked_n/N_TESTS*100]
colors_bar = ['#ff7f0e', '#2ca02c', '#1f77b4']
ax.bar(labels, values, color=colors_bar, edgecolor='white', width=0.5)
for i, v in enumerate(values):
    ax.text(i, v+2, f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_title('Phase-wise Accuracy')
ax.set_ylim(0, 115); ax.grid(True, alpha=0.3, axis='y')

# 6. Timing stats
ax = axes[1, 2]
ax.axis('off')
stats_text = (
    f"TEST REPORT\n"
    f"{'='*30}\n"
    f"Total tests:    {N_TESTS}\n"
    f"Total time:     {elapsed:.0f}s\n"
    f"Avg/test:       {elapsed/N_TESTS:.2f}s\n\n"
    f"Phase 1 (L/R):  {p1_ok/N_TESTS*100:.0f}%\n"
    f"Phase 2 (F/B):  {p2_ok/N_TESTS*100:.0f}%\n"
    f"Phase 3 (Lock): {locked_n/N_TESTS*100:.0f}%\n\n"
    f"Avg chase time: {np.mean(chase_t):.2f}s\n" if chase_t else "N/A\n"
    f"Min chase:      {np.min(chase_t):.2f}s\n" if chase_t else ""
    f"Max chase:      {np.max(chase_t):.2f}s\n" if chase_t else ""
    f"\nQuadrant detail:\n"
)
for qname, (lo, hi) in quadrants.items():
    br = [r for r in results if lo <= r['az'] < hi]
    if br:
        ok = sum(1 for r in br if r.get('locked'))
        stats_text += f"  {qname}: {ok}/{len(br)} ({ok/len(br)*100:.0f}%)\n"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Active Binaural Localization — Phase 1+2+3 Integrated Test',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'D:/shengxuedingwei2/results/phase3_report.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
