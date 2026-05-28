"""Phase 1+2+3 integrated test — 5 rounds of random source positions."""
import numpy as np
import yaml, tempfile, os, sys, time
sys.path.insert(0, 'D:/shengxuedingwei2')
from src.pipeline.simulator import BinauralSimulator
from src.localization.active_locator import ActiveLocator

N_ROUNDS = 5
N_PER_ROUND = 10  # 50 total

np.random.seed(2024)
all_results = []

for rnd in range(N_ROUNDS):
    print(f'\n=== Round {rnd+1}/{N_ROUNDS} ===')
    round_times = []; round_locked = 0
    for i in range(N_PER_ROUND):
        az = np.random.uniform(-180, 180)
        dist = np.random.uniform(1.5, 3.0)
        rad = np.deg2rad(az)
        px = float(4 + dist * np.sin(rad))
        py = float(4 + dist * np.cos(rad))

        cfg = {
            'name': f't{rnd}_{i}', 'description': '',
            'room': {'dimensions': [8, 8, 3], 'absorption': 1.0, 'max_order': 0},
            'microphone': {'head_center': [4, 4, 1.5], 'head_radius': 0.09, 'hrtf_mode': 'analytical'},
            'source': {'generator': 'human_voice', 'duration': 1.2, 'sample_rate': 44100,
                       'F0': 150, 'formants': [850, 1700, 2600, 3600]},
            'motion': {'enabled': False, 'type': 'static', 'position': [px, py, 1.5]},
            'output': {'sample_rate': 44100, 'visualize': False},
            'noise': {'enabled': False},
            'localization': {'active_head': True}, 'evaluation': {'enabled': False},
        }

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(cfg, tmp); tmp.close()
        t0 = time.time()
        try:
            sim = BinauralSimulator(tmp.name)
            stereo, _, _, _, _ = sim.run()
            loc = ActiveLocator(44100)
            result = loc.localize(stereo)
            elapsed = time.time() - t0

            locked = result.lock_achieved
            lock_at = result.lock_frame
            chase_s = ((lock_at - 12) * 0.025) if lock_at >= 0 else None
            if locked: round_locked += 1
            round_times.append(elapsed)

            all_results.append({
                'az': az, 'dist': dist, 'locked': locked,
                'chase_time': chase_s, 'total_time': elapsed,
            })
        except Exception as e:
            round_times.append(time.time() - t0)
            all_results.append({'az': az, 'dist': dist, 'locked': False, 'error': str(e)})
            print(f'  FAIL az={az:.0f}: {e}')
        finally:
            os.unlink(tmp.name)

    avg_t = np.mean(round_times)
    print(f'  Avg time: {avg_t:.2f}s | Locked: {round_locked}/{N_PER_ROUND}')

# Summary
locked_all = sum(1 for r in all_results if r['locked'])
chase_times = [r['chase_time'] for r in all_results if r['locked'] and r['chase_time'] is not None]
total_times = [r['total_time'] for r in all_results if 'total_time' in r]

print(f'\n======= FINAL REPORT =======')
print(f'Total tests: {len(all_results)}')
print(f'Lock success: {locked_all}/{len(all_results)} ({locked_all/len(all_results)*100:.0f}%)')
print(f'Avg total time: {np.mean(total_times):.2f}s')
print(f'Avg chase time (to lock): {np.mean(chase_times):.2f}s' if chase_times else 'N/A')
print(f'Min chase: {np.min(chase_times):.2f}s' if chase_times else '')
print(f'Max chase: {np.max(chase_times):.2f}s' if chase_times else '')

# By quadrant
for qname, mask in [('FR', lambda a: 0<=a<=90), ('FL', lambda a: -90<=a<0),
                     ('BR', lambda a: 90<a<=180), ('BL', lambda a: -180<=a<-90)]:
    br = [r for r in all_results if mask(r['az'])]
    if br:
        ok = sum(1 for r in br if r['locked'])
        print(f'  {qname}: {ok}/{len(br)} ({ok/len(br)*100:.0f}%)')
