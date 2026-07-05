import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Model imports
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Selection and metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Data Preparation ==========================================
# Load Training Data
df = pd.read_csv("training_data_ht2025.csv")

# Define feature groups
cat_features = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
num_features = ['temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']

# Map target to binary
df['target'] = df['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# 2. Exploratory Data Analysis (EDA) ============================

# Fig 1: Hourly Demand
plt.figure(figsize=(10, 5))
hourly = df.groupby('hour_of_day')['target'].mean() * 100
plt.plot(hourly.index, hourly.values, marker='o', color='navy')
plt.title('Probability of "High Demand" by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Probability (%)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.savefig('fig1_hourly_demand.png') # Saved for report
plt.show()

# Fig 2: Weekly Demand
plt.figure(figsize=(8, 5))
weekly = df.groupby('day_of_week')['target'].mean()
sns.barplot(x=weekly.index, y=weekly.values, color='orange')
plt.title('High demand rate by day of week')
plt.xlabel('Day of Week (0-6)')
plt.savefig('fig2_weekly_demand.png') # Saved for report
plt.show()

# Fig 3: Monthly Demand
plt.figure(figsize=(8, 5))
monthly = df.groupby('month')['target'].mean()
sns.barplot(x=monthly.index, y=monthly.values, color='forestgreen')
plt.title('High demand rate by month')
plt.xlabel('Month (1-12)')
plt.savefig('fig3_monthly_demand.png') # Saved for report
plt.show()

# Fig 4: Weekday vs Holiday
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

wd_prop = (
        df.groupby('weekday')['target']
              .value_counts(normalize=True)
                    .unstack(fill_value=0)
)
wd_prop.plot(kind='bar', stacked=True, ax=axes[0], legend=False)
axes[0].set_title('Weekday')
axes[0].set_xlabel('weekday (0=weekend, 1=weekday)')
axes[0].set_ylabel('Proportion')
axes[0].tick_params(axis='x', labelrotation=0)

hol_prop = (
        df.groupby('holiday')['target']
              .value_counts(normalize=True)
                    .unstack(fill_value=0)
)
hol_prop.plot(kind='bar', stacked=True, ax=axes[1], legend=False)
axes[1].set_title('Holiday')
axes[1].set_xlabel('holiday (0=no, 1=yes)')
axes[1].tick_params(axis='x', labelrotation=0)
handles, _ = axes[0].get_legend_handles_labels()
labels = ['Low demand (0)',
          'High demand (1)']
          fig.legend(
                handles,
                    labels,
                        title='Target',
                            loc='upper right',
                                ncol=1
          )
          fig.suptitle('Proportion of Target by Weekday and Holiday', fontsize=14)
          plt.tight_layout(rect=[0, 0, 1, 0.9])
          plt.savefig('fig4_weekday_holiday.png') # Saved for report
          plt.show()

          # Fig 5: Correlation Matrix
          plt.figure(figsize=(10, 8))
          plot_cols = ['temp', 'humidity', 'precip', 'windspeed', 'cloudcover', 'visibility', 'target']
          corr_data = df[plot_cols].rename(columns={'target': 'increase_stock'})
          sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt=".3f", linewidths=1, linecolor='white')
          plt.title('Correlation Matrix', fontweight='bold')
          plt.savefig('fig5_correlation.png') # Saved for report
          plt.show()

          # 3. Preprocessing =========================================

          # Drop "snow" because of zero variance in training set
          if 'snow' in df.columns:
              df = df.drop(columns=['snow'])

              # One-Hot Encoding for categorical proxies
              cat_cols_to_encode = ['day_of_week', 'month']
              df_processed = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True)

              # Define Features and Target
              X = df_processed.drop(columns=['increase_stock', 'target'])
              y = df_processed['target']

              # 80/20 Train-Validation Split
              X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

              # Scaling numerical features
              scaler = StandardScaler()
              X_train_scaled = scaler.fit_transform(X_train)
              X_val_scaled = scaler.transform(X_val)

              print("\n--- Model Benchmarking and Tuning ---")

              # 4. Model Training & Evaluation ==========================================

              # --- 1. Naive Baseline ---
              dummy = DummyClassifier(strategy="most_frequent")
              dummy.fit(X_train, y_train)
              val_f1_naive = f1_score(y_val, dummy.predict(X_val), average='weighted')
              print(f"Naive Benchmark Weighted F1: {val_f1_naive:.4f}")

              # --- 2. Logistic Regression ---
              # Pipeline setup
              log_pipe = Pipeline([
                    ('scaler', StandardScaler()), 
                        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
              ])

              # TUNING (Commented out)
              # param_grid_log = {'clf__C': [0.01, 0.1, 1, 10, 100]}
              # grid_log = GridSearchCV(log_pipe, param_grid_log, cv=5, scoring='f1_weighted')
              # grid_log.fit(X_train, y_train)
              # best_log = grid_log.best_estimator_
              # print("Best LogReg Params:", grid_log.best_params_)

              # Hardcoded Best Model
              log_model = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42)
              log_model.fit(X_train_scaled, y_train)
              pred_log = log_model.predict(X_val_scaled)

              print(f"Logistic Regression Weighted F1: {f1_score(y_val, pred_log, average='weighted'):.4f}")

              # --- 3. Linear Discriminant Analysis (LDA) ---
              lda = LinearDiscriminantAnalysis()
              lda.fit(X_train_scaled, y_train)
              pred_lda = lda.predict(X_val_scaled)
              print(f"LDA Weighted F1: {f1_score(y_val, pred_lda, average='weighted'):.4f}")

              # --- 4. Quadratic Discriminant Analysis (QDA) ---
              # QDA with all features
              qda = QuadraticDiscriminantAnalysis(reg_param=0.0001)
              qda.fit(X_train_scaled, y_train)
              pred_qda = qda.predict(X_val_scaled)
              print(f"QDA (All Features) Weighted F1: {f1_score(y_val, pred_qda, average='weighted'):.4f}")

              # QDA without 'dew'
              dew_col_idx = list(X.columns).index('dew') if 'dew' in X.columns else -1
              if dew_col_idx != -1:
                  X_train_no_dew = np.delete(X_train_scaled, dew_col_idx, axis=1)
                      X_val_no_dew = np.delete(X_val_scaled, dew_col_idx, axis=1)
                          
                              qda_no_dew = QuadraticDiscriminantAnalysis(reg_param=0.0001)
                                  qda_no_dew.fit(X_train_no_dew, y_train)
                                      pred_qda_no_dew = qda_no_dew.predict(X_val_no_dew)
                                          print(f"QDA (No Dew) Weighted F1: {f1_score(y_val, pred_qda_no_dew, average='weighted'):.4f}")

                                          # --- 5. K-Nearest Neighbors (KNN) ---
                                          # TUNING (Commented out)
                                          # knn_grid = GridSearchCV(
                                            #     KNeighborsClassifier(),
                                            #     param_grid={'n_neighbors': [3, 5, 10, 15], 'weights': ['uniform', 'distance']},
                                            #     cv=5,
                                            #     scoring='f1_weighted'
                                            # )
                                            # knn_grid.fit(X_train_scaled, y_train)
                                            # print(f"KNN Best Params: {knn_grid.best_params_}")

                                            # Hardcoded Best
                                            knn_best = KNeighborsClassifier(n_neighbors=10, weights='distance')
                                            knn_best.fit(X_train_scaled, y_train)
                                            pred_knn = knn_best.predict(X_val_scaled)
                                            print(f"KNN (Tuned) Weighted F1: {f1_score(y_val, pred_knn, average='weighted'):.4f}")

                                            # --- 6. Random Forest ---
                                            # TUNING (Commented out)
                                            # rf_params = {
                                                #     'n_estimators': [100, 200, 300],
                                                #     'max_depth': [None, 10, 20],
                                                #     'min_samples_split': [2, 5],
                                                #     'class_weight': ['balanced', None]
                                                # }
                                                # grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1_weighted', n_jobs=-1)
                                                # grid_rf.fit(X_train_scaled, y_train)
                                                # print(f"Random Forest Best Params: {grid_rf.best_params_}")

                                                # Hardcoded Best
                                                rf_best = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)
                                                rf_best.fit(X_train_scaled, y_train)
                                                pred_rf = rf_best.predict(X_val_scaled)
                                                print(f"Random Forest (Tuned) Weighted F1: {f1_score(y_val, pred_rf, average='weighted'):.4f}")


                                                # ==========================================
                                                # 7. Final Production Model: XGBoost
                                                # ==========================================
                                                print("\n--- Training Final Production Model (XGBoost) ---")

                                                # --- TUNING & VALIDATION STRATEGY (Commented out) ---
                                                # The specific parameters (n_estimators=2396) were derived using Early Stopping.
                                                #
                                                # 1. Grid Search for structural params (depth, learning_rate):
                                                # xgb_grid = GridSearchCV(
                                                    #     XGBClassifier(n_estimators=1000, tree_method='hist'),
                                                    #     param_grid = {
                                                        #         'max_depth': [6, 8, 10],
                                                        #         'learning_rate': [0.01, 0.05],
                                                        #         'scale_pos_weight': [1.0, 1.03] # Accounting for slight imbalance
                                                        #     },
                                                        #     scoring='f1_weighted', cv=3
                                                        # )
                                                        # xgb_grid.fit(X_train_scaled, y_train)
                                                        # best_params = xgb_grid.best_params_
                                                        #
                                                        # 2. Determine optimal n_estimators with Early Stopping:
                                                        # model_es = XGBClassifier(**best_params, n_estimators=5000, random_state=42)
                                                        # model_es.fit(X_train_scaled, y_train, 
                                                        #              eval_set=[(X_val_scaled, y_val)], 
                                                        #              early_stopping_rounds=50, verbose=False)
                                                        # print(f"Optimal n_estimators: {model_es.best_iteration + 1}") 
                                                        # # Result -> 2396 trees
                                                        
                                                        # Exact parameters derived from the strategy above
                                                        xgb_production_params = {
                                                                "n_estimators": 2396,          # Found via early stopping
                                                                    "learning_rate": 0.01,
                                                                        "max_depth": 8,
                                                                            "scale_pos_weight": 1.03,      # Tuned for class balance
                                                                                "eval_metric": "logloss",
                                                                                    "random_state": 42,
                                                                                        "tree_method": "hist"
                                                        }

                                                        final_model = XGBClassifier(**xgb_production_params)
                                                        final_model.fit(X_train_scaled, y_train)

                                                        # Evaluation
                                                        y_pred_final = final_model.predict(X_val_scaled)
                                                        print("\n--- Final Production Model Report (XGBoost) ---")
                                                        print(classification_report(y_val, y_pred_final))
                                                        print(f"Final Weighted F1 Score: {f1_score(y_val, y_pred_final, average='weighted'):.4f}")

                                                        # --- Visualizations ---

                                                        # 1. Confusion Matrix
                                                        plt.figure(figsize=(6, 5))
                                                        sns.heatmap(confusion_matrix(y_val, y_pred_final), annot=True, fmt='d', cmap='Blues')
                                                        plt.title("Confusion Matrix: Gradient Boosting")
                                                        plt.xlabel("Predicted")
                                                        plt.ylabel("Actual")
                                                        plt.tight_layout()
                                                        plt.savefig('confusion_matrix_xgb.png')
                                                        print("Confusion Matrix saved to 'confusion_matrix_xgb.png'")

                                                        # 2. Top Features Importance
                                                        importances = final_model.feature_importances_
                                                        feature_names = list(X.columns)
                                                        feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                                                        feat_df = feat_df.sort_values(by='importance', ascending=False).head(10)

                                                        plt.figure(figsize=(10, 6))
                                                        sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
                                                        plt.title("Top 10 Feature Importance (XGBoost)")
                                                        plt.tight_layout()
                                                        plt.savefig('feature_importance_xgb.png')
                                                        print("Feature Importance saved to 'feature_importance_xgb.png'")

                                                        # 8. Submission File Generation
                                                        TEST_FILE_NAME = "test_data_fall2025.csv"

                                                        try:
                                                            print(f"\n--- Generating Predictions for {TEST_FILE_NAME} ---")
                                                                df_test = pd.read_csv(TEST_FILE_NAME)
                                                                    
                                                                        # --- Preprocessing Test Set ---
                                                                            # 1. Drop snow
                                                                                if 'snow' in df_test.columns:
                                                                                        df_test = df_test.drop(columns=['snow'])
                                                                                            
                                                                                                # 2. One-Hot Encode (Align columns with training)
                                                                                                    df_test_processed = pd.get_dummies(df_test, columns=cat_cols_to_encode, drop_first=True)
                                                                                                        
                                                                                                            # 3. Align columns (Add missing cols with 0, drop extra cols)
                                                                                                                # Get missing columns in the test set
                                                                                                                    missing_cols = set(X.columns) - set(df_test_processed.columns)
                                                                                                                        for c in missing_cols:
                                                                                                                                df_test_processed[c] = 0
                                                                                                                                    # Reorder columns to match training X
                                                                                                                                        df_test_processed = df_test_processed[X.columns]
                                                                                                                                            
                                                                                                                                                # 4. Scale
                                                                                                                                                    X_test_scaled = scaler.transform(df_test_processed)
                                                                                                                                                        
                                                                                                                                                            # --- Prediction ---
                                                                                                                                                                predictions = final_model.predict(X_test_scaled)
                                                                                                                                                                    
                                                                                                                                                                        # --- Formatting for Submission ---
                                                                                                                                                                            submission_string = ",".join(map(str, predictions))
                                                                                                                                                                                
                                                                                                                                                                                    # Save to CSV
                                                                                                                                                                                        with open("predictions.csv", "w") as f:
                                                                                                                                                                                                f.write(submission_string)
                                                                                                                                                                                                        
                                                                                                                                                                                                            print("Success! 'predictions.csv' has been created.")
                                                                                                                                                                                                                print(f"Prediction count: {len(predictions)}")
                                                                                                                                                                                                                    print(f"Sample: {submission_string[:20]}...")

                                                                                                                                                                                                                    except FileNotFoundError:
                                                                                                                                                                                                                        print(f"WARNING: Test file '{TEST_FILE_NAME}' not found.")
                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                            print(f"Error: {e}")
                                                        })
                                                    }
                                                )
                                            }
                                          )
              ])
          )]
)
)