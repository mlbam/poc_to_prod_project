import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_train_samples = MagicMock(return_value=80)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_test_samples = MagicMock(return_value=20)
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        index_to_label = {0: 'a', 1: 'b', 2: 'c'}
        self.assertEqual(base.get_index_to_label_map(), index_to_label)

    def test_index_to_label_and_label_to_index_are_identity(self):
        # define 2 dictionaries
        index_to_label = {0: 'a', 1: 'b', 2: 'c'}
        label_to_index = {'a': 0, 'b': 1, 'c': 2}
        # check if there are identity
        self.assertEqual(index_to_label, {v: k for k, v in label_to_index.items()})
        self.assertEqual(label_to_index, {v: k for k, v in index_to_label.items()})
        # check if they are inverse of each other
        self.assertEqual(index_to_label, {label_to_index[k]: k for k in label_to_index})
        self.assertEqual(label_to_index, {index_to_label[k]: k for k in index_to_label})


    def test_to_indexes(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        labels = ['a', 'b', 'c']
        indexes = [0, 1, 2]
        self.assertEqual(base.to_indexes(labels), indexes)

class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2','id_3'],
            'tag_name': ['tag_a', 'tag_b','tag_a'],
            'tag_id': [1, 2, 3],
            'tag_position': [0, 1, 0],
            'title': ['title_1', 'title_2','title_3']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, train_ratio=0.5, min_samples_per_label=1)
        self.assertEqual(dataset._get_num_samples(), 2)

    def test_get_train_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3'],
            'tag_name': ['tag_a', 'tag_a', 'tag_a'],
            'tag_id': [1, 2, 3],
            'tag_position': [0, 1, 0],
            'title': ['title_1', 'title_2', 'title_2']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, train_ratio=0.5, min_samples_per_label=1)
        self.assertEqual(len(dataset.get_train_batch()[0]), 1)
        self.assertEqual(len(dataset.get_train_batch()[1]), 1)

    def test_get_test_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a'],
            'tag_id': [1, 2, 3],
            'tag_position': [0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, train_ratio=0.5, min_samples_per_label=1)
        self.assertEqual(len(dataset.get_test_batch()[0]), 1)
        self.assertEqual(len(dataset.get_test_batch()[1]), 1)

    def test_get_train_batch_raises_assertion_error(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a'],
            'tag_id': [1, 2, 3],
            'tag_position': [1, 1, 1],
            'title': ['title_1', 'title_2', 'title_3']
        }))

        # we expect an assertion error to be raised
        with self.assertRaises(AssertionError):
            # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
            dataset = utils.LocalTextCategorizationDataset("fake_path", 1, train_ratio=0.5, min_samples_per_label=1)
            dataset.get_train_batch()



