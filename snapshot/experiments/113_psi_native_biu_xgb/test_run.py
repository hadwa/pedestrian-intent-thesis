import ast
import unittest
from types import SimpleNamespace

import numpy as np
import run


class Tests(unittest.TestCase):
    def test_native_vote_tie_and_uncertainty(self):
        native = run.module('psi113_test_votes', run.ROOT / 'official_code/baseline/data/process_sequence.py')
        ann = {str(i): dict(intent=values, description=[''] * 3) for i, values in enumerate([
            ['not_sure', 'cross', 'not_cross'], ['not_sure', 'not_cross', 'not_sure']])}
        db = {'v': {'p': {'frames': [1, 2, 3], 'nlp_annotations': ann}}}
        labels, votes, _, _ = native.get_intent(db, 'v', 'p', SimpleNamespace(intent_type='mean', intent_num=2))
        self.assertEqual(labels, [1, 1, 0])
        np.testing.assert_array_equal(votes, [.5, .5, .25])

    def test_native_window_sampling(self):
        # Execute only the pinned pure sampler, avoiding its unrelated image-loader imports.
        path = run.ROOT / 'official_code/baseline/data/prepare_data.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'get_tracks')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        data = {k: [list(range(30))] for k in ('video_id', 'ped_id', 'frame', 'bbox', 'intention_binary', 'intention_prob', 'disagree_score', 'description')}
        for overlap in (.9, 1):
            windows = namespace['get_tracks'](data, 16, 15, overlap, SimpleNamespace(max_track_size=16))
            np.testing.assert_array_equal(windows['frame'], [list(range(s, s + 16)) for s in range(15)])

    def test_no_future_and_dimensions(self):
        xy = np.arange(15 * 17 * 2, dtype=float).reshape(15, 17, 2) + 1
        conf = np.ones((15, 17))
        boxes = np.tile([0., 0., 100., 200.], (15, 1))
        values, _, _ = run.extract(xy, conf, boxes)
        self.assertEqual([len(values[a]) for a in ('A', 'B', 'C')], [53, 125, 116])
        np.testing.assert_array_equal(values['A'], values['B'][:53])
        np.testing.assert_array_equal(values['A'], values['C'][:53])
        with self.assertRaises(AssertionError):
            run.extract(np.zeros((16, 17, 2)), np.ones((16, 17)), np.ones((16, 4)))

    def test_test_annotation_guard(self):
        with self.assertRaises(RuntimeError):
            open('/tmp/PSI2.0_Test/annotations/example.json')


if __name__ == '__main__':
    unittest.main()
