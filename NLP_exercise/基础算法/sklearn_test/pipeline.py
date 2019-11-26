import numpy as np

from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    Nystroem(gamma=10.0, kernel="polynomial", n_components=10),
    make_union(VotingClassifier([("est", KNeighborsClassifier(n_neighbors=4, weights="distance"))]), FunctionTransformer(lambda X: X)),
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="entropy", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    FeatureAgglomeration(affinity="precomputed", linkage="average"),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
