# THis file is so we can add all the required inputs for generating pipelines easily,
# without cluttering the code with all these imports

# We need all of these for generating pipelines
from sklearn.pipeline import Pipeline
from src.feature_preprocessors import SelectMutualInfo, SelectFClassif, SelectChi2
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.similarity_based.fisher_score import fisher_score
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import Imputer
from src.sklearn_additions import StackingCV