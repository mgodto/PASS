import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def format_classification_report_percent(y_true, y_pred, class_names):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy_score_val = report_dict.pop('accuracy')

    df = pd.DataFrame(report_dict).transpose()

    def fmt_pct(x):
        return f"{x:.2%}"

    for col in ['precision', 'recall', 'f1-score']:
        df[col] = df[col].apply(fmt_pct)

    df['support'] = df['support'].apply(lambda x: str(int(x)))

    total_support = df.loc['macro avg', 'support']
    accuracy_row = pd.DataFrame(
        {
            'precision': [''],
            'recall': [''],
            'f1-score': [fmt_pct(accuracy_score_val)],
            'support': [total_support],
        },
        index=['accuracy'],
    )

    avgs = df.loc[['macro avg', 'weighted avg']]
    classes = df.drop(['macro avg', 'weighted avg'])

    final_df = pd.concat([classes, accuracy_row, avgs])

    return final_df.to_string(), accuracy_score_val


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(8, 6)):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_ylabel('Actual Class')
    ax.set_xlabel('Predicted Class')
    ax.set_title(title)
    return fig
