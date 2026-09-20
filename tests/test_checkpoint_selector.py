import csv
from pathlib import Path

from scripts import checkpoint_selector as selector


FIELDS = (
    'episode', 'episode_step', 'is_last', 'success', 'collision', 'timeout',
    'minimum_clearance')


def write_scene(path, outcomes):
    path.mkdir(parents=True)
    with (path / 'trajectory_actions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for episode, outcome in enumerate(outcomes, 1):
            writer.writerow({
                'episode': episode, 'episode_step': 0, 'is_last': 0,
                'success': 0, 'collision': 0, 'timeout': 0,
                'minimum_clearance': 10.0})
            writer.writerow({
                'episode': episode, 'episode_step': 10, 'is_last': 1,
                'success': int(outcome == 'success'),
                'collision': int(outcome == 'collision'),
                'timeout': int(outcome == 'timeout'),
                'minimum_clearance': 0.25 + episode / 100})
        # A partially started episode must not affect the summary.
        writer.writerow({
            'episode': len(outcomes) + 1, 'episode_step': 0, 'is_last': 0,
            'success': 0, 'collision': 0, 'timeout': 0,
            'minimum_clearance': 0.01})


def test_scene_metrics_uses_only_complete_episodes(tmp_path):
    scene = tmp_path / 'scene'
    write_scene(scene, ['success', 'success', 'collision', 'timeout'])
    metrics = selector.scene_metrics(scene)
    assert metrics['episodes'] == 4
    assert metrics['successes'] == 2
    assert metrics['collisions'] == 1
    assert metrics['timeouts'] == 1
    assert metrics['success_rate'] == 0.5
    assert metrics['minimum_clearance'] == 0.26
    assert metrics['mean_navigation_time'] == 1.0


def test_retention_uses_one_episode_review_band():
    def metrics(rate, episodes):
        return {'success_rate': rate, 'episodes': episodes}
    parent = {
        'fixed0p8': metrics(0.88, 100),
        'fixed0p6': metrics(0.89, 100),
    }
    candidate = {
        'fixed0p8': metrics(0.80, 20),
        'fixed0p6': metrics(0.80, 20),
    }
    status, checks = selector.retention(candidate, parent)
    assert status == 'REVIEW'
    assert checks['fixed0p8']['threshold'] == 0.83
    assert checks['fixed0p6']['threshold'] == 0.84
    assert checks['fixed0p6']['episode_resolution'] == 0.05


def test_retention_fails_outside_review_band():
    parent = {
        name: {'success_rate': 0.90, 'episodes': 100}
        for name in ('fixed0p8', 'fixed0p6')}
    candidate = {
        name: {'success_rate': 0.75, 'episodes': 20}
        for name in ('fixed0p8', 'fixed0p6')}
    status, _ = selector.retention(candidate, parent)
    assert status == 'FAIL'


def test_ranking_prioritizes_retention_without_weighted_score():
    def entry(screening, random_success):
        scenes = {}
        for name in selector.SCENES:
            scenes[name] = {
                'success_rate': random_success if name == 'random' else 0.8,
                'collision_rate': 0.0,
                'timeout_rate': 0.2,
                'mean_minimum_clearance': 0.3,
                'mean_navigation_time': 10.0,
            }
        return {'screening': screening, 'metrics': {'scenes': scenes}}
    passed = entry('PASS', 0.6)
    review = entry('REVIEW', 1.0)
    assert selector.rank_key(passed) > selector.rank_key(review)
