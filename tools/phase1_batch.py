"""Phase 1 batch test — 100 random positions, left/right detection accuracy."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml, tempfile, os, sys, time
sys.path.insert(0, 'D:/shengxuedingwei2')
from src.pipeline.simulator import BinauralSimulator
from src.localization.active_locator import ActiveLocator

np.random.seed(42)
n_tests = 100
results = []
print(f'Running {n_tests} tests...')
t0 = time.time()

for i in range(n_tests):
    az = np.random.uniform(-180, 180)
    dist = np.random.uniform(1.0, 3.0)
    rad = np.deg2rad(az)
    px = float(4.0 + dist * np.sin(rad))
    py = float(4.0 + dist * np.cos(rad))

    cfg = {
        'name': f't{i}', 'description': '',
        'room': {'dimensions': [8, 8, 3], 'absorption': 1.0, 'max_order': 0},
        'microphone': {'head_center': [4, 4, 1.5], 'head_radius': 0.09, 'hrtf_mode': 'analytical'},
        'source': {'generator': 'human_voice', 'duration': 0.6, 'sample_rate': 44100,
                   'F0': 150, 'formants': [850, 1700, 2600, 3600]},
        'motion': {'enabled': False, 'type': 'static', 'position': [px, py, 1.5]},
        'output': {'sample_rate': 44100, 'visualize': False},
        'noise': {'enabled': False}, 'localization': {}, 'evaluation': {'enabled': False},
    }

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp); tmp.close()
    try:
        sim = BinauralSimulator(tmp.name)
        stereo, _, _, _, _ = sim.run()
        loc = ActiveLocator(44100)
        source_left = loc._phase1_detect_side(stereo)
        true_left = az < 0
        correct = (source_left == true_left)
        results.append({'az': az, 'dist': dist, 'correct': correct})
    except Exception as e:
        results.append({'az': az, 'dist': dist, 'correct': False})
    finally:
        os.unlink(tmp.name)

    if (i+1) % 10 == 0:
        c = sum(1 for r in results if r['correct'])
        print(f'  {i+1}/{n_tests}: {c}/{i+1} ({c/(i+1)*100:.0f}%)')

correct_n = sum(1 for r in results if r['correct'])
print(f'\nAccuracy: {correct_n}/{n_tests} ({correct_n/n_tests*100:.1f}%)')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for r in results:
    c = 'g' if r['correct'] else 'r'
    ax.scatter(r['az'], r['dist'], c=c, s=30, alpha=0.7, edgecolors='none')
ax.axvline(x=0, c='w', ls='--', lw=1)
ax.set_xlabel('True Azimuth (deg)'); ax.set_ylabel('Distance (m)')
ax.set_title(f'Phase 1 L/R Detection ({correct_n}/{n_tests})')
ax.set_xlim(-185, 185); ax.grid(True, alpha=0.3)

ax2 = axes[1]
for lo in np.arange(-180, 180, 20):
    hi = lo + 20
    br = [r for r in results if lo <= r['az'] < hi]
    if br:
        acc = sum(1 for r in br if r['correct']) / len(br) * 100
        ax2.bar((lo+hi)/2, acc, width=18, color='steelblue', edgecolor='white')
ax2.axhline(y=100, c='g', ls='--', lw=1)
ax2.set_xlabel('True Azimuth (deg)'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy by Angle Bin'); ax2.set_ylim(0, 110); ax2.grid(True, alpha=0.3)

plt.suptitle('Phase 1 Left/Right Detection — 100 Random Tests', fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'D:/shengxuedingwei2/results/phase1_results.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
