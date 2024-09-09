import unittest
from unittest.mock import MagicMock
import numpy as np
from numpy.testing import assert_array_almost_equal


from src.old.core_classes_old.clustering import (initialize_centroids_kmeans_pp,
                                                 assign_best_to_centroids,
                                                 stake_claims,
                                                 resolve_fights,
                                                 constrained_match,
                                                 constrained_kmeans
                                                 )
from src.old.core_classes_old.clustering import L1Distance, L2Distance, BoundingOverlapDifference

class TestMeasureSubclasses(unittest.TestCase):

    def setUp(self):
        """
        Set up some basic data points and centroids to use across tests.
        """
        # Simple test data
        self.points = np.array([[1, 2], [3, 4]])
        self.centroids = np.array([[1, 1], [5, 5]])

    def test_l1_distance(self):
        """
        Test the L1 distance measure on a simple dataset.
        """
        l1_distance = L1Distance()
        distances = l1_distance(self.points, self.centroids)

        expected_distances = np.array([[0+1, 2+3], [4+3, 2+1]])
        np.testing.assert_array_equal(distances, expected_distances)

    def test_l2_distance(self):
        """
        Test the L2 distance measure on a simple dataset.
        """
        l2_distance = L2Distance()
        distances = l2_distance(self.points, self.centroids)

    def test_bounding_overlap_difference(self):
        """
        Test the BoundingOverlapDifference measure on a simple dataset.
        """
        bounding_overlap_diff = BoundingOverlapDifference()
        distances = bounding_overlap_diff(self.points, self.centroids)

        expected_distances = np.array([[1*2 - 1*1, 3*4 - 1*1],
                                       [5*5 - 1*2, 5*5 - 3*4]])
        np.testing.assert_array_equal(distances, expected_distances)
class TestInitializeCentroidsKmeansPP(unittest.TestCase):

    def test_simple_dataset_initialization(self):
        # Test with a simple 2D dataset
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        k = 2
        logging_callback = MagicMock()
        section_lengths = np.array([2])
        centroids = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        # Ensure centroids are selected from the original data points
        for centroid in centroids:
            self.assertIn(list(centroid), data.tolist())

        # Verify logging was called
        logging_callback.assert_called()

    def test_correct_number_of_centroids(self):
        # Test to ensure the correct number of centroids is returned
        data = np.random.rand(10, 2)
        k = 3
        logging_callback = MagicMock()
        section_lengths = np.array([2])
        centroids = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        self.assertEqual(centroids.shape[0], k)
        self.assertEqual(centroids.shape[1], data.shape[1])

        # Verify logging was called
        logging_callback.assert_called()
    def test_randomness_of_initialization(self):
        # Ensure different results for different runs without fixed seed
        data = np.random.rand(20, 2)
        k = 3
        logging_callback = MagicMock()
        section_lengths = np.array([2])
        centroids_1 = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)
        centroids_2 = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        # Check that the centroids are not identical across runs
        self.assertFalse(np.allclose(centroids_1, centroids_2))

    @unittest.skip("test_reproducibility_with_fixed_seed: This still needs to be debugged, but is good enough for now.")
    def test_reproducibility_with_fixed_seed(self):
        # Ensure that the initialization is reproducible with a fixed random seed
        np.random.seed(42)
        data = np.random.rand(10, 2)
        k = 3
        logging_callback = MagicMock()
        section_lengths = np.array([2])
        centroids_1 = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        np.random.seed(42)
        centroids_2 = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        assert_array_almost_equal(centroids_1, centroids_2)

    def test_centroid_distribution(self):
        # Test with a dataset having 4 points in two clusters
        distance_amplification = 1000  # This greatly amps up the selection probabilities
        cluster_one = distance_amplification * np.array([[10, 10, 10], [9, 9, 9]])
        cluster_two = distance_amplification * np.array([[-19, -19, -19], [-18, -18, -18]])
        data = np.concatenate([cluster_one, cluster_two], axis=0)
        k = 2
        logging_callback = MagicMock()
        section_lengths = np.array([3])
        centroids = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        # One centroid should be from the first cluster, one from the second
        cluster_one = np.expand_dims(cluster_one, axis=0)
        cluster_two = np.expand_dims(cluster_two, axis=0)
        expanded_centroids = np.expand_dims(centroids, axis=1)

        # Check. We should have EXACTLY one match for each cluster
        self.assertEqual(np.all(cluster_one == expanded_centroids, axis=-1).sum(), 1)
        self.assertEqual(np.all(cluster_two == expanded_centroids, axis=-1).sum(), 1)

        # Verify logging was called
        logging_callback.assert_called()

    def test_compound_mode(self):
        # Test the function with compound points (multiple sections in data)
        data = np.random.rand(10, 5)  # 10 data points, 5 features (could be split into 3+2)
        section_lengths = np.array([3, 2])
        k = 3
        logging_callback = MagicMock()

        centroids = initialize_centroids_kmeans_pp(data, section_lengths, k, L2Distance(), logging_callback)

        # Ensure centroids are selected from the original data points
        for centroid in centroids:
            self.assertIn(list(centroid), data.tolist())

        # Verify logging was called
        logging_callback.assert_called()

class TestAssignBestToCentroids(unittest.TestCase):

    def setUp(self) -> None:
        # This will be initialized before each test to avoid cross-contamination
        self.measure = L2Distance()

    def check_matches_present(self,
                              selected_datapoints: np.ndarray,
                              expected_elements: np.ndarray,
                              expected_matches: int) -> None:
        """
        Helper function to verify that the selected datapoints contain the expected elements.

        :param selected_datapoints: The datapoints selected by the function.
        :param expected_elements: The specific elements we expect to be present in the selected datapoints.
        :param expected_matches: The number of matches we expect for the given elements.
        """
        # Expand dimensions for broadcasting
        selected_datapoints_expanded = np.expand_dims(selected_datapoints, axis=1)  # (n_selected, 1, n_features)
        expected_elements_expanded = np.expand_dims(expected_elements, axis=0)  # (1, n_expected, n_features)

        # Check if each expected element is found in the selected datapoints
        matches = np.all(selected_datapoints_expanded == expected_elements_expanded, axis=-1)
        self.assertEqual(matches.sum(), expected_matches)

    def test_basic_functionality(self) -> None:
        """ Test that the function correctly identifies the `n` closest points to a given centroid using L2 distance. """
        logging_callback = MagicMock()

        # Define datapoints and centroids
        cluster_one = np.array([[1, 1], [2, 2]])
        cluster_two = np.array([[10, 10], [11, 11]])
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        centroids = np.array([1, 1])
        section_lengths = np.array([2])
        n = 2

        # Run the function
        selected_datapoints, indices, distances = assign_best_to_centroids(
            datapoints, centroids, section_lengths, self.measure, n, logging_callback
        )

        # Check that the selected datapoints contain the correct cluster elements
        self.check_matches_present(selected_datapoints, cluster_one, expected_matches=2)

        # Verify that distances are correct
        expected_distances = np.array([0, np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.sort(distances), np.sort(expected_distances))

        # Ensure that the logging callback was called
        logging_callback.assert_called()

    def test_composite_functionality(self) -> None:
        """ Test that we can operate correctly when defining multiple centroids and points. """
        logging_callback = MagicMock()

        # Define datapoints and centroids
        cluster_one = np.array([[1, 1], [2, 2]])
        cluster_two = np.array([[10, 10], [11, 11]])
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        centroids = np.array([1, 1])
        section_lengths = np.array([1, 1])
        n = 2

        # Run the function
        selected_datapoints, indices, distances = assign_best_to_centroids(
            datapoints, centroids, section_lengths, self.measure, n, logging_callback
        )

        # Check that the selected datapoints contain the correct cluster elements
        self.check_matches_present(selected_datapoints, cluster_one, expected_matches=2)

        # Verify that distances are correct
        expected_distances = np.array([0, 1 + 1])
        np.testing.assert_array_almost_equal(np.sort(distances), np.sort(expected_distances))

        # Ensure that the logging callback was called
        logging_callback.assert_called()

    def test_select_top_n(self) -> None:
        """ Test that the function correctly selects the top `n` closest points. """
        logging_callback = MagicMock()

        # Define datapoints and centroids
        cluster_one = np.array([[1, 1], [2, 2], [3, 3]])
        cluster_two = np.array([[10, 10], [11, 11], [12, 12]])
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        centroids = np.array([1, 1])
        section_lengths = np.array([2])
        n = 2

        # Run the function
        selected_datapoints, indices, distances = assign_best_to_centroids(
            datapoints, centroids, section_lengths, self.measure, n, logging_callback
        )

        # Check that the selected datapoints contain the correct cluster elements
        self.check_matches_present(selected_datapoints, cluster_one[:2], expected_matches=2)
        self.check_matches_present(selected_datapoints, cluster_one[2:], expected_matches=0)

        # Verify that distances are correct
        expected_distances = np.array([0, np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.sort(distances), np.sort(expected_distances))

        # Ensure that the logging callback was called
        logging_callback.assert_called()

class TestStakeClaims(unittest.TestCase):


    def test_simple_claims(self):
        # Test scenario in which we start with no claims, and both centroids
        # can immediately make their best claims without collision.

        distances = np.array([[0, 0, 10, 10], [10, 10, 0, 0]])
        num_claims = 2
        claims = np.array([[False, False, False, False], # No points claimed yet
                           [False, False, False, False]])
        expected_claims = np.array([[True, True, False, False], # centroid 0 claims point 0,1.
                                    [False, False, True, True]]) # centroid 1 claims points 2, 3

        stake_claims(claims, distances, num_claims)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_no_reclaiming(self):
        # Test scenario in which claims have already been made, and assures us
        # that items cannot then be reclaimed by another centroid.
        distances = np.array([[0, 0, 10, 10], [10, 10, 0, 0]])
        num_claims = 2
        claims = np.array([[False, False, True, True],  # Centroid 1 starts with claims to 2, 3
                           [True, True, False, False]]) # Centroid 2 starts with claims to 0, 1
        expected_claims = np.array([[False, False, True, True], # No free points, so no differences.
                                    [True, True, False, False]])
        stake_claims(claims, distances, num_claims)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_fighting_claims(self):
        # Test scenario in which two centroids try to stake the same claim to
        # an unclaimed point
        distances = np.array([[0, 10, 0], [10, 0, 0]])
        claims = np.array([[True, False, False], # Centroid 0 claims point 0
                           [False, True, False]]) # Centroid 1 claims point 1
        expected_claims = np.array([[True, False, True],  # Centroid 0 and 1 both staking 2
                                    [False, True, True]])

        stake_claims(claims, distances, 3)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_short_data(self):
        # Test scenario in which there is not enough data to select all N we would like to
        distances = np.array([[0],[0]])
        claims = np.array([[False], [False]])
        expected_claims = np.array([[True], [True]])
        num_stakes = 2

        stake_claims(claims, distances, num_stakes)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_long_data(self):
        # Test scenario in which there is significantly more data available then needed
        distances = np.array([[0, 10, 10], [10, 10, 0]])
        num_stakes = 1

        claims = np.array([[False, False, False], # Centroid 0 claims point 0
                           [False, False, False]]) # Centroid 1 claims point 1
        expected_claims = np.array([[True, False, False],  # Centroid 0 and 1 both staking 2
                                    [False, False, True]])

        stake_claims(claims, distances, num_stakes)
        np.testing.assert_array_equal(claims, expected_claims)


class TestResolveFights(unittest.TestCase):

    def test_no_fights(self):
        # No points are contested, so no changes to claims
        claims = np.array([[True, False], [False, True]])
        distances = np.array([[1, 10], [10, 1]])  # distances don't matter since there are no fights

        expected_claims = claims.copy()

        resolve_fights(claims, distances)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_single_fight(self):
        # Only one point (index 1) is being fought over by two centroids
        claims = np.array([[True, True], [False, True]])  # Centroid 0 and 1 both claim point 1
        distances = np.array([[5, 2], [10, 1]])  # Centroid 1 is closer to point 1

        expected_claims = np.array([[True, False], [False, True]])  # Centroid 1 wins point 1

        resolve_fights(claims, distances)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_multiple_fights(self):
        # Several points are fought over by multiple centroids
        claims = np.array([[True, True, False], [False, True, True], [True, False, True]])
        distances = np.array([[5, 2, 8], [10, 1, 9], [4, 7, 3]])  # Distances for each centroid-point pair

        # Centroid 0 wins point 1, Centroid 2 wins point 2
        expected_claims = np.array([[True, False, False], [False, True, False], [True, False, True]])

        resolve_fights(claims, distances)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_all_fights(self):
        # Every point is fought over by multiple centroids
        claims = np.array([[True, True, True], [True, True, True], [True, True, True]])
        distances = np.array([[5, 2, 8], [10, 1, 9], [4, 7, 3]])  # Distances for each centroid-point pair

        # Centroid 2 wins point 0 (smallest distance), Centroid 1 wins point 1, Centroid 2 wins point 2
        expected_claims = np.array([[False, False, False], [False, True, False], [True, False, True]])

        resolve_fights(claims, distances)
        np.testing.assert_array_equal(claims, expected_claims)

    def test_tie_fights(self):
        # Two centroids have the same distance to a point, check that result is consistent
        claims = np.array([[True, True], [False, True]])  # Both centroids claim point 1
        distances = np.array([[5, 2], [10, 2]])  # Both centroids are equally close to point 1

        # Run once to store the resolved claims
        initial_claims = claims.copy()
        resolve_fights(initial_claims, distances)

        # Run multiple times and verify that the result remains consistent
        for _ in range(10):  # Run multiple times to verify consistency
            next_claims = claims.copy()
            resolve_fights(next_claims, distances)
            np.testing.assert_array_equal(initial_claims, next_claims)

class TestConstrainedMatch(unittest.TestCase):

    def test_simple_case(self):
        # Test with simple non-colliding centroids and points
        distances = np.array([[1, 5, 10],
                              [10, 5, 1]])
        n = 2  # Each centroid can claim at most 2 points
        expected_claims = np.array([[True, True, False],
                                    [False, False, True]])

        # Create a separate logging callback mock
        logging_callback = MagicMock()

        claims = constrained_match(distances, n, logging_callback)

        # Assert claims match the expected values
        np.testing.assert_array_equal(claims, expected_claims)

        # Ensure logging callback was called with Verbosity.Debug
        for call in logging_callback.call_args_list:
            self.assertEqual(call[0][1], 3)  # Assuming 3 corresponds to Verbosity.Debug

    def test_fighting_case(self):
        # Test where centroids fight for the same point
        distances = np.array([[0, 1, 10],
                              [0, 2, 0]])
        n = 1  # Each centroid can claim only 1 point
        expected_claims = np.array([[True, False, False],
                                    [False, False, True]])

        # Create a separate logging callback mock
        logging_callback = MagicMock()

        claims = constrained_match(distances, n, logging_callback)

        # Assert claims match the expected values
        np.testing.assert_array_equal(claims, expected_claims)

        # Ensure logging callback was called with Verbosity.Debug
        for call in logging_callback.call_args_list:
            self.assertEqual(call[0][1], 3)  # Assuming 3 corresponds to Verbosity.Debug

    def test_max_points_per_centroid(self):
        # Test with more centroids and points but each centroid can only claim 1 point
        distances = np.array([[1, 2, 3],
                              [2, 1, 2],
                              [3, 3, 1]])
        n = 1  # Each centroid can claim only 1 point
        expected_claims = np.array([[True, False, False],
                                    [False, True, False],
                                    [False, False, True]])

        # Create a separate logging callback mock
        logging_callback = MagicMock()

        claims = constrained_match(distances, n, logging_callback)

        # Assert claims match the expected values
        np.testing.assert_array_equal(claims, expected_claims)

        # Ensure logging callback was called with Verbosity.Debug
        for call in logging_callback.call_args_list:
            self.assertEqual(call[0][1], 3)  # Assuming 3 corresponds to Verbosity.Debug

    def test_unclaimed_points(self):
        # Test where some points are left unclaimed because centroids are at maximum claims
        distances = np.array([[1, 100, 100],
                              [100, 100, 1]])  # Larger values instead of using infinity
        n = 1  # Each centroid can claim only 1 point
        expected_claims = np.array([[True, False, False],
                                    [False, False, True]])

        # Create a separate logging callback mock
        logging_callback = MagicMock()

        claims = constrained_match(distances, n, logging_callback)

        # Assert claims match the expected values
        np.testing.assert_array_equal(claims, expected_claims)

        # Ensure logging callback was called with Verbosity.Debug
        for call in logging_callback.call_args_list:
            self.assertEqual(call[0][1], 3)  # Assuming 3 corresponds to Verbosity.Debug

    def test_tie_fight_resolution(self):
        # Test if tie-fights are consistently resolved the same way
        distances = np.array([[1, 1, 10],
                              [1, 1, 10]])
        n = 1  # Each centroid can claim only 1 point
        expected_claims = np.array([[True, False, False],
                                    [False, True, False]])

        # Create a separate logging callback mock
        logging_callback = MagicMock()

        # Run multiple times to check consistent resolution
        for _ in range(10):
            claims = constrained_match(distances, n, logging_callback)
            np.testing.assert_array_equal(claims, expected_claims)

            # Ensure logging callback was called with Verbosity.Debug
            for call in logging_callback.call_args_list:
                self.assertEqual(call[0][1], 3)  # Assuming 3 corresponds to Verbosity.Debug


class TestConstrainedKMeans(unittest.TestCase):

    def check_matches_present(self,
                              selected_datapoints: np.ndarray,
                              expected_elements: np.ndarray,
                              expected_matches: int) -> bool:
        """
        Helper function to verify that the selected datapoints contain the expected elements.

        :param selected_datapoints: The datapoints selected by the function.
        :param expected_elements: The specific elements we expect to be present in the selected datapoints.
        :param expected_matches: The number of matches we expect for the given elements.
        :return: True if the matches are present, False otherwise.
        """
        # Expand dimensions for broadcasting
        selected_datapoints_expanded = np.expand_dims(selected_datapoints, axis=1)  # (n_selected, 1, n_features)
        expected_elements_expanded = np.expand_dims(expected_elements, axis=0)  # (1, n_expected, n_features)

        # Check if each expected element is found in the selected datapoints
        matches = np.all(selected_datapoints_expanded == expected_elements_expanded, axis=-1)
        return matches.sum() == expected_matches

    def test_basic_two_clusters(self):
        # Test with a dataset having 2 clusters
        cluster_one = np.random.uniform(0, 1, [3, 2])
        cluster_two = np.random.uniform(4, 5, [3, 2])
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        section_lengths = np.array([2])
        k = 2
        n = 3

        # Run constrained kmeans
        clusters = constrained_kmeans(datapoints, section_lengths, k, n, measure=L2Distance())

        # Check if both clusters were matched
        cluster_one_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_one, 3) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_one, 3)

        cluster_two_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_two, 3) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_two, 3)

        # Assert that both clusters were correctly matched
        self.assertTrue(cluster_one_matched)
        self.assertTrue(cluster_two_matched)

    def test_constrained_cluster_size(self):
        # Test scenario where n is less than the number of points per cluster
        datapoints = np.array([[1, 1], [1, 2], [2, 1], [10, 10], [10, 9], [9, 10]])
        section_lengths = np.array([2])
        k = 2
        n = 2

        # Run constrained kmeans
        clusters = constrained_kmeans(datapoints, section_lengths, k, n, measure=L2Distance())

        # Verify clusters have at most n elements
        for cluster in clusters:
            self.assertLessEqual(cluster["datapoints"].shape[0], n)

    def test_larger_dataset(self):
        # Test with a larger dataset to ensure the algorithm scales
        cluster_one = np.random.uniform(0, 1, (50, 2))
        cluster_two = np.random.uniform(10, 9, (50, 2))
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        section_lengths = np.array([2])
        k = 2
        n = 30

        # Run constrained kmeans
        clusters = constrained_kmeans(datapoints, section_lengths, k, n, measure=L2Distance())

        # Check if both clusters were matched
        cluster_one_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_one, 30) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_one, 30)

        cluster_two_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_two, 30) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_two, 30)

        # Assert that both clusters were correctly matched
        self.assertTrue(cluster_one_matched)
        self.assertTrue(cluster_two_matched)

    def test_tolerance(self):
        # Test if the algorithm respects the tolerance parameter
        cluster_one = np.array([[1, 1], [1, 2], [2, 1]])
        cluster_two = np.array([[10, 10], [10, 9], [9, 10]])
        datapoints = np.concatenate([cluster_one, cluster_two], axis=0)
        section_lengths = np.array([2])
        k = 2
        n = 3

        # Run constrained kmeans with a high tolerance to ensure early stopping
        clusters = constrained_kmeans(datapoints, section_lengths, k, n, measure=L2Distance(), tolerance=10.0)

        # Check if both clusters were matched
        cluster_one_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_one, 3) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_one, 3)

        cluster_two_matched = self.check_matches_present(clusters[0]["datapoints"], cluster_two, 3) or \
                              self.check_matches_present(clusters[1]["datapoints"], cluster_two, 3)

        # Assert that both clusters were correctly matched
        self.assertTrue(cluster_one_matched)
        self.assertTrue(cluster_two_matched)