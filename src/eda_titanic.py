import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Data loading & paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic.csv')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Q1-a: exploratory data analysis
def basic_overview(df):
    print("Shape:", df.shape)
    print("\nColumns and types:\n", df.dtypes)
    display_df = df.head(8)
    print("\nPreview:\n", display_df.to_string(index=False))
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nNumeric summary:\n", df.describe(include='all').T)

def plot_target_distribution(df):
    fig, ax = plt.subplots()
    counts = df['Survived'].value_counts().sort_index()
    counts.plot(kind='bar', ax=ax)
    ax.set_xticklabels(['Died (0)', 'Survived (1)'])
    ax.set_ylabel('Count')
    ax.set_title('Survival distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'survival_distribution.png'))
    plt.close(fig)

def plot_categorical_count(df, col, title=None, filename=None):
    fig, ax = plt.subplots()
    df[col].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(title or f'Count: {col}')
    ax.set_ylabel('Count')
    fig.tight_layout()
    if filename:
        fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)

def plot_hist_numeric(df, col, bins=30, filename=None, by_survived=False):
    if by_survived:
        fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
        for i, surv in enumerate(sorted(df['Survived'].unique())):
            subset = df[df['Survived']==surv][col].dropna()
            axes[i].hist(subset, bins=bins)
            axes[i].set_title(f"{col} distribution (Survived={surv})")
            axes[i].set_xlabel(col)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
    if filename:
        fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)

def boxplot_by_target(df, numeric_col, by='Survived', filename=None):
    fig, ax = plt.subplots()
    df.boxplot(column=numeric_col, by=by, ax=ax)
    ax.set_title(f"{numeric_col} by {by}")
    ax.set_ylabel(numeric_col)
    plt.suptitle('')
    fig.tight_layout()
    if filename:
        fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)

def correlation_and_heatmap(df, encode=True, filename='corr_heatmap.png'):
    df_corr = df.copy()
    if encode:
        # Encode Sex
        if 'Sex' in df_corr.columns:
            df_corr['Sex_enc'] = LabelEncoder().fit_transform(df_corr['Sex'].astype(str))
        # One-hot for Embarked and Pclass or label encode
        if 'Embarked' in df_corr.columns:
            df_corr['Embarked_enc'] = LabelEncoder().fit_transform(df_corr['Embarked'].astype(str))
        if 'Pclass' in df_corr.columns:
            df_corr['Pclass_enc'] = df_corr['Pclass']  # already numeric usually
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_corr[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation matrix (numeric features)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    return corr

def survival_rates_by_group(df, group_col, filename=None):
    grp = df.groupby(group_col)['Survived'].agg(['count', 'mean']).reset_index()
    grp.columns = [group_col, 'count', 'survival_rate']
    fig, ax = plt.subplots()
    ax.bar(grp[group_col].astype(str), grp['survival_rate'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Survival Rate')
    ax.set_title(f'Survival rate by {group_col}')
    fig.tight_layout()
    if filename:
        fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    return grp

def feature_engineering_family(df):
    df = df.copy()
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def main():
    df = load_data()
    basic_overview(df)

    # plots & stats
    plot_target_distribution(df)
    plot_categorical_count(df, 'Sex', title='Count by Sex', filename='count_sex.png')
    plot_categorical_count(df, 'Pclass', title='Count by Pclass', filename='count_pclass.png')
    if 'Embarked' in df.columns:
        plot_categorical_count(df, 'Embarked', title='Count by Embarked', filename='count_embarked.png')

    # numeric
    if 'Age' in df.columns:
        plot_hist_numeric(df, 'Age', bins=30, filename='hist_age.png', by_survived=False)
        boxplot_by_target(df, 'Age', filename='box_age_by_survived.png')
    if 'Fare' in df.columns:
        plot_hist_numeric(df, 'Fare', bins=30, filename='hist_fare.png', by_survived=False)
        boxplot_by_target(df, 'Fare', filename='box_fare_by_survived.png')

    # engineered features
    df_fe = feature_engineering_family(df)
    if 'FamilySize' in df_fe.columns:
        survival_rates_by_group(df_fe, 'FamilySize', filename='survival_by_family_size.png')
        survival_rates_by_group(df_fe, 'IsAlone', filename='survival_by_isalone.png')

    # correlation heatmap
    corr = correlation_and_heatmap(df)

    # survival rates for Sex and Pclass
    sr_sex = survival_rates_by_group(df, 'Sex', filename='survival_by_sex.png')
    sr_pclass = survival_rates_by_group(df, 'Pclass', filename='survival_by_pclass.png')

    # Save small CSV tables of grouped stats
    sr_sex.to_csv(os.path.join(FIG_DIR, 'survival_by_sex.csv'), index=False)
    sr_pclass.to_csv(os.path.join(FIG_DIR, 'survival_by_pclass.csv'), index=False)
    print("Figures saved in:", FIG_DIR)

if __name__ == "__main__":
    main()