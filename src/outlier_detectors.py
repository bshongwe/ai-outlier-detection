"""Outlier detection algorithms for AI Outlier Detection Pipeline."""

import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Base class for outlier detection algorithms."""

    def __init__(self, name: str):
        self.name = name

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the given DataFrame."""
        raise NotImplementedError


class EuclideanDetector(OutlierDetector):
    """Euclidean distance-based outlier detection."""

    def __init__(self, radius: float = 0.55):
        super().__init__("Euclidean")
        self.radius = radius

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Euclidean distance from class centroids."""
        logger.info(f"Running {self.name} outlier detection with radius {self.radius}")

        centroids = self._get_centroids(df)
        outlier_indices = []

        for idx, row in df.iterrows():
            class_name = row["Class Name"]
            distance = self._euclidean_distance(row["Embeddings"], centroids[class_name])

            if distance > self.radius:
                outlier_indices.append(idx)

        df[f'Outlier_{self.name}'] = False
        df.loc[outlier_indices, f'Outlier_{self.name}'] = True

        logger.info(f"Found {len(outlier_indices)} outliers using {self.name} method")
        return df

    def _get_centroids(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate centroids for each class."""
        centroids = {}
        for class_name, group in df.groupby("Class Name"):
            centroids[class_name] = np.mean(np.vstack(group['Embeddings']), axis=0)
        return centroids

    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors."""
        return np.sqrt(np.sum(np.square(p1 - p2)))


class MahalanobisDetector(OutlierDetector):
    """Mahalanobis distance-based outlier detection."""

    def __init__(self, confidence_level: float = 0.99):
        super().__init__("Mahalanobis")
        self.confidence_level = confidence_level

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Mahalanobis distance."""
        logger.info(f"Running {self.name} outlier detection with confidence {self.confidence_level}"
    )

        outlier_indices = []

        for class_name, group in df.groupby("Class Name"):
            embeddings = np.vstack(group['Embeddings'])

            if len(embeddings) < 2:
                continue

            mean = np.mean(embeddings, axis=0)
            cov = np.cov(embeddings.T)

            try:
                inv_cov = np.linalg.inv(cov)
                threshold = chi2.ppf(self.confidence_level, df=embeddings.shape[1])

                for idx in group.index:
                    embedding = df.loc[idx, 'Embeddings']
                    distance = mahalanobis(embedding, mean, inv_cov)

                    if distance**2 > threshold:
                        outlier_indices.append(idx)

            except np.linalg.LinAlgError:
                logger.warning(f"Singular covariance matrix for class {class_name}")
                continue

        df[f'Outlier_{self.name}'] = False
        df.loc[outlier_indices, f'Outlier_{self.name}'] = True

        logger.info(f"Found {len(outlier_indices)} outliers using {self.name} method")
        return df


class LOFDetector(OutlierDetector):
    """Local Outlier Factor-based outlier detection."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        super().__init__("LOF")
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Local Outlier Factor."""
        logger.info(f"Running {self.name} outlier detection with {self.n_neighbors} neighbors")

        embeddings = np.vstack(df['Embeddings'])

        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )

        outlier_labels = lof.fit_predict(embeddings)
        df[f'Outlier_{self.name}'] = outlier_labels == -1

        outlier_count = sum(outlier_labels == -1)
        logger.info(f"Found {outlier_count} outliers using {self.name} method")

        return df


class IsolationForestDetector(OutlierDetector):
    """Isolation Forest-based outlier detection."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        super().__init__("IsolationForest")
        self.contamination = contamination
        self.random_state = random_state

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using Isolation Forest."""
        logger.info(f"Running {self.name} outlier detection with contamination {self.contamination}"
    )

        embeddings = np.vstack(df['Embeddings'])

        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )

        outlier_labels = iso_forest.fit_predict(embeddings)
        df[f'Outlier_{self.name}'] = outlier_labels == -1

        outlier_count = sum(outlier_labels == -1)
        logger.info(f"Found {outlier_count} outliers using {self.name} method")

        return df


class OutlierDetectionPipeline:
    """Orchestrates multiple outlier detection algorithms."""

    def __init__(self, detectors: List[OutlierDetector]):
        self.detectors = detectors

    def run_all_detectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all configured outlier detection algorithms."""
        logger.info(f"Running {len(self.detectors)} outlier detection algorithms")

        result_df = df.copy()

        for detector in self.detectors:
            result_df = detector.detect(result_df)

        return result_df

    def get_consensus_outliers(self, df: pd.DataFrame, min_detectors: int = 2) -> List[int]:
        """Get outliers detected by multiple algorithms."""
        outlier_columns = [col for col in df.columns if col.startswith('Outlier_')]

        consensus_outliers = []
        for idx, row in df.iterrows():
            detection_count = sum(row[col] for col in outlier_columns)
            if detection_count >= min_detectors:
                consensus_outliers.append(idx)

        logger.info(f"Found {len(consensus_outliers)} consensus outliers")
        return consensus_outliers
