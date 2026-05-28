"""100 random source position tests for Phase 1 left/right detection."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml, tempfile, os, time
from src.pipeline.simulator import BinauralSimulator
from src.localization.active_locator import ActiveLocator

np.random.seed(42)
n_tests = 100
results = []

print(f'Running {n_tests} random position tests...')
t0 = time.time()

for i in range(n_tests):
    az = np.random.uniform(-180, 180)  # random azimuth
    dist = np.random.uniform(1.0, 3.0)  # random distance
    rad = np.deg2rad(az)

    # Build scene config
    cfg = {
        'name': f'test_{i}',
        'description': '',
        'room': {'dimensions': [8, 8, 3], 'absorption': 1.0, 'max_order': 0},
        'microphone': {'head_center': [4.0, 4.0, 1.5], 'head_radius': 0.09,
                       'hrtf_mode': 'parametric'},
        'source': {'generator': 'human_voice', 'duration': 0.8, 'sample_rate': 44100,
                   'F0': 150, 'formants': [850, 1700, 2600, 3600]},
        'motion': {'enabled': False, 'type': 'static',
                   'position': [4.0 + dist*np.sin(rad), 4.0 + dist*np.cos(rad), 1.5]},
        'output': {'sample_rate': 44100, 'visualize': False},
        'noise': {'enabled': False},
        'localization': {'method': 'gcc_phat'},
        'evaluation': {'enabled': False},
    }

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp); tmp.close()

    try:
        sim = BinauralSimulator(tmp.name)
        stereo, _, _, _, _ = sim.run()
        loc = ActiveLocator(44100)
        result = loc.localize(stereo)
        detected_left = result.method  # stored in method for now
        # Actually check the Phase 1 output by re-running detection
        source_left = loc._phase1_detect_side(stereo)
        true_left = az < 0
        correct = (source_left == true_left)
        results.append({'az': az, 'dist': dist, 'correct': correct,
                        'detected': 'L' if source_left else 'R',
                        'true': 'L' if true_left else 'R'})
    except Exception as e:
        results.append({'az': az, 'dist': dist, 'correct': False, 'error': str(e)})
    finally:
        os.unlink(tmp.name)

    if (i+1) % 20 == 0:
        elapsed = time.time() - t0
        correct_so_far = sum(1 for r in results if r.get('correct'))
        print(f'  {i+1}/{n_tests}: {correct_so_far}/{i+1} correct ({correct_so_far/(i+1)*100:.0f}%), {elapsed:.0f}s')

# Statistics
correct = [r for r in results if r.get('correct')]
errors = [r for r in results if not r.get('correct') and 'error' not in r]
print(f'\nResults: {len(correct)}/{len(results)} correct ({len(correct)/len(results)*100:.1f}%)')
if errors:
    print(f'Errors: {len(errors)} frames')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax = axes[0]
for r in results:
    color = 'green' if r.get('correct') else 'red'
    marker = 'o'
    ax.scatter(r['az'], r['dist'], c=color, marker=marker, s=30, alpha=0.7,
               edgecolors='none')
ax.axvline(x=0, color='white', linestyle='--', linewidth=1)
ax.set_xlabel('True Azimuth (deg)')
ax.set_ylabel('Distance (m)')
ax.set_title(f'Phase 1: Left/Right Detection ({len(correct)}/{n_tests} correct)')
ax.set_xlim(-185, 185)
ax.set_ylim(0.5, 3.5)
ax.grid(True, alpha=0.3)

# Confusion by angle
ax2 = axes[1]
bins = np.arange(-180, 181, 30)
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    bin_results = [r for r in results if lo <= r['az'] < hi]
    if bin_results:
        acc = sum(1 for r in bin_results if r.get('correct')) / len(bin_results) * 100
        ax2.bar((lo+hi)/2, acc, width=25, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axhline(y=100, color='green', linestyle='--', linewidth=1)
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('True Azimuth (deg)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy by Angle Bin')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3)

plt.suptitle('Phase 1 Left/Right Detection — 100 Random Tests', fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = 'D:/shengxuedingwei2/results/phase1_test.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nFigure saved: {out_path}')
plt.close()
