
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class_weight = ['balanced', None]
n_jobs = 1
random_state = 42

rf_random_grid = {'bootstrap': [True, False],
                  'max_depth': [10, 50, 100, None],
                  'max_features': ['auto', 'log2', None],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [1000, 2000],
                  'class_weight': class_weight+["balanced_subsample"]}

lr_random_grid = {'C' : np.logspace(-3, 2, 6),
                  'penalty' : ['l2', 'none'],
                  'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
                  'class_weight' : class_weight}
				  'class_weight' : class_weight}

classifiers = {
    "lr" : {"clf" : LogisticRegression(n_jobs=n_jobs, random_state=random_state), "random_grid" : lr_random_grid, "clf_with_params" : LogisticRegression(n_jobs=n_jobs, random_state=random_state)},
    "rf" : {"clf" : RandomForestClassifier(n_jobs=n_jobs, random_state=random_state), "random_grid" : rf_random_grid, "clf_with_params" : RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)}
}

# Select the features
issue_features = [        
    'num_comments', 'num_events', 'commits_count', 'is_pull_request', 'num_of_assignees', 'has_milestone',   
    'cm_mean_len', 'time_to_discuss', 'cm_developers_ratio',    
    'body_processed_len', 'title_processed_len', 'title_processed_words_num', 'body_processed_words_num', 
    'title_alphabet_ratio', 'body_alphabet_ratio',   
    'num_of_codesnippets',
    'num_of_functions',
    'num_of_issues',
    'num_of_paths',
    'num_of_urls',    
    'ft_bug',
    'ft_feature',       
    'body_sentistrenght_p',
    'title_subjectivity',
    'body_subjectivity',
    'positive_body_sentistrenght_n',
    'positive_title_polarity',
    'positive_body_polarity',
]   
user_features = [
    'author_followers', 'author_following', 'author_public_repos', 'author_public_gists', 'author_issue_counts', 
    'author_github_cntrb', 'author_repo_cntrb', 'author_account_age', 'numeric_association'
]
all_features = issue_features + user_features

# Classify
def classify(dataset, algorithm=, param_mode="default", target_column="priority", save_importances=True):        
    df = dataframes[dataset]
    
    X_train = df[df.test_tag == 0][all_features]
    X_test = df[df.test_tag == 1][all_features]
    y_train = df[df.test_tag == 0][target_column]
    y_test = df[df.test_tag == 1][target_column]

    title = f"{dataset}_{param_mode}_{algorithm}_{target_column}"    
    report = title + ":\n"

    if param_mode == "default":
        model = classifiers[algorithm]["clf"]
    elif param_mode == "specified":
        model = classifiers[algorithm]["clf_with_params"]
    elif param_mode == "tuned":
        model = RandomizedSearchCV(estimator=classifiers[algorithm]["clf"], 
									param_distributions = classifiers[algorithm]["random_grid"], 
									n_iter=100, cv=3, random_state=42, n_jobs=-1)
        

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report += classification_report(y_test, y_pred)
    if(param_mode == "tuned"):
        report += "\nbestparameters:\n" + str(model.best_params_) + '\n'
    accuracy = accuracy_score(y_pred, y_test)
    report += "\naccuracy score:" + str(accuracy) + '\n'
    with open(f"results/{title}.txt", "w") as f:
        f.write(report)
    print(report)

    if algorithm == "rf" and save_importances:
        nfeatures  = X_test.shape[1]
        fig, ax = plt.subplots(dpi=300, figsize = [20,15])
        ax.barh(range(nfeatures), model.feature_importances_)
        ax.set_yticks(range(nfeatures))
        ax.set_yticklabels(all_features)
        fig.savefig(f"results/images/{title}")

# Retrieve the target repositories
repos_list = [] #populate with the names of target repositories
dataframes = {}
for repo in repos_list:
    dataframes[repo] = pd.read_csv(f"../results/{repo}_norm.csv")
	

# Run the models	
algorithm = "rf"
param_mode = "default"
target_column = "priority" 
save_importances = False
for repo in repos_list:
    classify(repo, algorithm, param_mode, target_column, save_importances)

