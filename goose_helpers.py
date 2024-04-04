import math
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from prettytable import PrettyTable


def display_search_results(search_estimator, X_train=None, y_train=None, X_val=None, y_val=None):
    """
    Display hyperparameter tuning results and classification reports for training and validation datasets.

    Summarizes the results of hyperparameter tuning from a trained GridSearchCV
    or RandomizedSearchCV for classification models. If training data (`X_train` and `y_train`)
    is provided, it displays the classification report for the training dataset. If validation data
    (`X_val` and `y_val`) is provided, it also displays the classification report for the validation dataset.

    Parameters
    ----------
    search_estimator : sklearn estimator
        Trained instance of GridSearchCV or RandomizedSearchCV.
    X_train : array-like, shape (n_samples, n_features), optional
        Training dataset to compute training metrics. Default is None.
    y_train : array-like, shape (n_samples, ), optional
        Actual labels for `X_train`. Used only if `X_train` is provided. Default is None.
    X_val : array-like, shape (n_samples, n_features), optional
        Validation dataset to compute additional metrics. Default is None.
    y_val : array-like, shape (n_samples, ), optional
        Actual labels for `X_val`. Used only if `X_val` is provided. Default is None.

    Returns
    -------
    None
        Outputs the tuning results and classification reports to the console.

    Notes
    -----
    Assumes `search_estimator` is a fit instance of GridSearchCV or RandomizedSearchCV.
    The model's name is extracted from `search_estimator` for the results table title.
    """

    # Determine model name based on whether it's part of a pipeline
    if hasattr(search_estimator, 'estimator') and hasattr(search_estimator.estimator, 'steps'):
        # It's a pipeline
        model_name = search_estimator.estimator.steps[-1][1].__class__.__name__
    else:
        # Direct model
        model_name = search_estimator.best_estimator_.__class__.__name__

    title = f"{model_name} Tuning Results"

    # PrettyTable display for tuning results
    myTable = PrettyTable(["Parameter", "Value"])

    if hasattr(search_estimator, 'best_params_'):
        for key, value in search_estimator.best_params_.items():
            myTable.add_row([key, value])
    else:
        myTable.add_row(["No parameters found", "N/A"])

    myTable.title = title
    print(myTable)

    # Display classification report for the training set if provided
    if X_train is not None and y_train is not None:
        y_train_pred = search_estimator.predict(X_train)
        print("Classification Report - Training Set:")
        print(classification_report(y_train, y_train_pred))

    # Display classification report for the validation set if provided
    if X_val is not None and y_val is not None:
        y_val_pred = search_estimator.predict(X_val)
        print("Classification Report - Validation Set:")
        print(classification_report(y_val, y_val_pred))

def split_data(df, target_var, verbose=True):
    """
    Splits a DataFrame into training, validation, and test datasets.

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains all the data.
    target_var : str
        The column in the DataFrame that represents the target variable.
    verbose : bool, optional
        Whether or not to print a report of the split. Default is True.

    Returns
    -------
    X_train : DataFrame
        Feature data for training.
    y_train : Series
        Target data for training.
    X_val : DataFrame
        Feature data for validation.
    y_val : Series
        Target data for validation.
    X_test : DataFrame
        Feature data for testing.
    y_test : Series
        Target data for testing.
    X_train_val : DataFrame
        Combined feature data for training and validation.
    y_train_val : Series
        Combined target data for training and validation.
    """

    # Define features and target
    X = df.drop(target_var, axis=1)
    y = df[target_var]

    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Split train+val set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val)

    # Reset index for all datasets
    for data in [X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val]:
        data.reset_index(drop=True, inplace=True)

    if verbose:
        # Print sizes of split datasets
        print("Data split complete\n")
        print("Training set:", X_train.shape, y_train.shape)
        print("Validation set:", X_val.shape, y_val.shape)
        print("Test set:", X_test.shape, y_test.shape)
        print("Combined Training and Validation set:", X_train_val.shape, y_train_val.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val

def plot_categorical_features(df):
    """
    Plot count plots for categorical features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    list
        A list of categorical features in the DataFrame.
    """
    # Suppress user warnings and future warnings (palette without assigning hue)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set color palette
    custom_palette = sns.color_palette("muted")

    # Select object, category, and boolean columns, and numerical columns with fewer than 25 unique values
    categorical_features = df.select_dtypes(include=['object', "category", "bool"]).columns.tolist() + \
                           [col for col in df.select_dtypes(include=np.number).columns if df[col].nunique() <= 27]

    rows = math.ceil(len(categorical_features) / 3)  # Calculate the number of rows required

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, 3, figsize=(20, 5 * rows), sharey=True)

    for i, ax in zip(categorical_features, axs.flat):
        # Plotting
        sns_plot = sns.countplot(data=df, x=i, ax=ax, palette=custom_palette)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), ha='center', fontsize=16)

        ax.tick_params(axis='y', labelsize=16)  # Adjust y-tick label size

        # Annotate bars
        for p in sns_plot.patches:
            sns_plot.annotate(format(p.get_height(), '.0f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center',
                              va='center',
                              xytext=(0, 10),
                              textcoords='offset points',
                              fontsize=12)

        ax.set_title(i, fontsize=22)
        sns.despine()

    # Hide any unused subplots
    for ax in axs.flat[len(categorical_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    return categorical_features

def plot_continuous_features(df):
    """
    Plot histograms, stripplots, and boxplots for continuous features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    list
        A list of continuous features in the DataFrame.
    """
    # Set the color palette
    sns.set_palette(sns.color_palette(["#4796C8"]))

    continuous_features = []

    for i in df.select_dtypes('number').columns:
        if len(df[i].unique()) > 27:  # Condition for continuous features
            continuous_features.append(i)

            fig, ax = plt.subplots(1, 3, figsize=(20, 6))

            # Plots
            sns.histplot(data=df[i], ax=ax[0], kde=True, alpha=0.7)
            sns.stripplot(x=df[i], ax=ax[1], alpha=0.7)
            sns.boxplot(x=df[i], ax=ax[2])

            # Stripplot median and mean line
            median = df[i].median()
            mean = df[i].mean()
            ax[1].axvline(x=median, color='#6AA2D8', linestyle='-', linewidth=3, ymin=0.35, ymax=0.65, zorder=3)  # median
            ax[1].axvline(x=mean, color='#6AA2D8', linestyle='--', linewidth=3, ymin=0.35, ymax=0.65, zorder=3)   # mean

            # Title entire figure
            fig.suptitle(i, fontsize=18, y=1)

            # Hide ticks while keeping labels
            ax[0].tick_params(bottom=False)
            ax[1].tick_params(bottom=False)
            ax[2].tick_params(bottom=False)
            ax[0].tick_params(left=False)
            ax[1].tick_params(left=False)
            ax[2].tick_params(left=False)

            # Hide x and y labels
            ax[0].set(xlabel=None, ylabel=None)
            ax[1].set(xlabel=None, ylabel=None)
            ax[2].set(xlabel=None, ylabel=None)

            sns.despine()

    return continuous_features

def display_classification_report(model, X_train, y_train, X_val=None, y_val=None, data_to_display='train'):
    """
    Display classification report for specified data sets: training, validation, or both.

    Parameters
    ----------
    model : estimator
        Fitted classifier or pipeline.
    X_train : pd.DataFrame
        Training input samples.
    y_train : pd.Series
        Training target values.
    X_val : pd.DataFrame, optional
        Validation input samples. Required if data_to_display is 'validation' or 'both'.
    y_val : pd.Series, optional
        Validation target values. Required if data_to_display is 'validation' or 'both'.
    data_to_display : str, default='train'
        Which data's classification report to display: 'train', 'validation', or 'both'.
    """
    if data_to_display in ['train', 'both']:
        # Generate predictions for training set
        y_train_pred = model.predict(X_train)
        # Print classification report for training set
        print("Classification Report (Training):")
        print(classification_report(y_train, y_train_pred))

    if data_to_display in ['validation', 'both']:
        if X_val is not None and y_val is not None:
            # Generate predictions for validation set
            y_val_pred = model.predict(X_val)
            # Print classification report for validation set
            print("\nClassification Report (Validation):")
            print(classification_report(y_val, y_val_pred))
        else:
            print("Validation data not provided for requested 'validation' or 'both' reports.")

def display_confusion_matrix(model, X_train, y_train, X_val=None, y_val=None, data_to_display='train'):
    """
    Display confusion matrix for specified data sets: training, validation, or both.

    Parameters
    ----------
    model : estimator
        Fitted classifier or pipeline.
    X_train : pd.DataFrame
        Training input samples.
    y_train : pd.Series
        Training target values.
    X_val : pd.DataFrame, optional
        Validation input samples. Required if data_to_display is 'validation' or 'both'.
    y_val : pd.Series, optional
        Validation target values. Required if data_to_display is 'validation' or 'both'.
    data_to_display : str, default='train'
        Which data's confusion matrix to display: 'train', 'validation', or 'both'.
    """
    fig_size = (10, 5)

    if data_to_display in ['train', 'both']:
        # Generate and plot confusion matrix for training set
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)
        if data_to_display == 'train':
            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(ax=ax)
            ax.set_title('Confusion Matrix (Training)')
        else:  # If both, prepare subplot and plot training CM first
            fig, axes = plt.subplots(1, 2, figsize=(fig_size[0] * 2, fig_size[1]))
            ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(ax=axes[0])
            axes[0].set_title('Confusion Matrix (Training)')

    if data_to_display in ['validation', 'both']:
        if X_val is not None and y_val is not None:
            # Generate and plot confusion matrix for validation set
            y_val_pred = model.predict(X_val)
            cm_val = confusion_matrix(y_val, y_val_pred)
            if data_to_display == 'validation':
                fig, ax = plt.subplots(1, 1, figsize=fig_size)
                ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(ax=ax)
                ax.set_title('Confusion Matrix (Validation)')
            else:  # For 'both', plot validation CM in second subplot
                ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(ax=axes[1])
                axes[1].set_title('Confusion Matrix (Validation)')
        else:
            print("Validation data not provided for requested 'validation' or 'both' reports.")

    plt.tight_layout()
    plt.show()

def display_roc_curve(model, X_train, y_train, X_val=None, y_val=None, data_to_display='train'):
    """
    Display ROC curve for specified data sets: training, validation, or both.

    Parameters
    ----------
    model : estimator
        Fitted classifier or pipeline.
    X_train : pd.DataFrame
        Training input samples.
    y_train : pd.Series
        Training target values.
    X_val : pd.DataFrame, optional
        Validation input samples. Required if data_to_display is 'validation' or 'both'.
    y_val : pd.Series, optional
        Validation target values. Required if data_to_display is 'validation' or 'both'.
    data_to_display : str, default='train'
        Which data's ROC curve to display: 'train', 'validation', or 'both'.
    """
    if data_to_display == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        RocCurveDisplay.from_estimator(model, X_train, y_train, ax=axes[0])
        axes[0].set_title('ROC Curve (Training)')
        RocCurveDisplay.from_estimator(model, X_val, y_val, ax=axes[1])
        axes[1].set_title('ROC Curve (Validation)')
        plt.tight_layout()
        plt.show()

    elif data_to_display == 'train':
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax)
        ax.set_title('ROC Curve (Training)')
        plt.tight_layout()
        plt.show()

    elif data_to_display == 'validation':
        if X_val is not None and y_val is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax)
            ax.set_title('ROC Curve (Validation)')
            plt.tight_layout()
            plt.show()
        else:
            print("Validation data not provided for requested 'validation' report.")


