import optuna
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.cluster
from sklearn.metrics import accuracy_score
from optuna.visualization import plot_optimization_history, plot_slice, plot_parallel_coordinate

# Загрузка датасета Iris
dataset = sklearn.datasets.load_iris()
X = dataset.data
y = dataset.target

# Функция для классификации (Logistic Regression)
def objective_classification(trial):
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Определение гиперпараметров для Logistic Regression
    C = trial.suggest_loguniform("C", 1e-5, 1e2)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])

    classifier = sklearn.linear_model.LogisticRegression(C=C, solver=solver, random_state=42)
    classifier.fit(X_train, y_train)
    
    predictions = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy

# Функция для кластеризации (KMeans)
def objective_clustering(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    init = trial.suggest_categorical("init", ["k-means++", "random"])
    max_iter = trial.suggest_int("max_iter", 100, 500)

    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
    clusterer.fit(X)
    return clusterer.inertia_

# Настройка хранения в PostgreSQL
storage_url = "postgresql://postgres:1234@localhost:5432/laba"
study_name_classification = "iris_classification"
study_name_clustering = "iris_clustering"

# Создание и запуск изучения для классификации
study_classification = optuna.create_study(
    study_name=study_name_classification,
    storage=storage_url,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    direction="maximize",
    load_if_exists=False
)
study_classification.optimize(objective_classification, n_trials=50)

# Создание и запуск изучения для кластеризации
study_clustering = optuna.create_study(
    study_name=study_name_clustering,
    storage=storage_url,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    direction="minimize",
    load_if_exists=False
)
study_clustering.optimize(objective_clustering, n_trials=50)

# Вывод лучших параметров и значения для классификации
print("Лучшие параметры классификации:", study_classification.best_params)
print("Лучшее значение классификации:", study_classification.best_value)

# Вывод лучших параметров и значения для кластеризации
print("Лучшие параметры кластеризации:", study_clustering.best_params)
print("Лучшее значение кластеризации:", study_clustering.best_value)

# Визуализация результатов для классификации
plot_optimization_history(study_classification).show()
plot_slice(study_classification).show()
plot_parallel_coordinate(study_classification).show()

# Визуализация результатов для кластеризации
plot_optimization_history(study_clustering).show()
plot_slice(study_clustering).show()
plot_parallel_coordinate(study_clustering).show()
