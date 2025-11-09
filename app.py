import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Set page config first
st.set_page_config(
    page_title="Voting Classifier App",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Voting Classifier - Ensemble Learning")
st.markdown("Interactive app to compare Voting Classifier with base estimators")

# Dataset functions
def generate_datasets():
    np.random.seed(42)
    
    # Linearly separable
    X_linear, y_linear = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=2.0, random_state=42
    )
    
    # Circles
    X_circle, y_circle = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    
    # XOR
    X_xor = np.random.randn(300, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0).astype(int)
    
    # Moons (U-shape)
    from sklearn.datasets import make_moons
    X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    # Blobs with outliers
    from sklearn.datasets import make_blobs
    X_blob, y_blob = make_blobs(n_samples=300, centers=2, random_state=42)
    # Add outliers
    outlier_idx = np.random.choice(len(X_blob), 20, replace=False)
    X_blob[outlier_idx] += np.random.normal(0, 3, (20, 2))
    
    return {
        "Linearly separable": (X_linear, y_linear),
        "Concentric circles": (X_circle, y_circle),
        "XOR": (X_xor, y_xor),
        "U-shape (Moons)": (X_moon, y_moon),
        "Outlier": (X_blob, y_blob)
    }

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Dataset selection
datasets = generate_datasets()
dataset_name = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))

# Classifier selection
st.sidebar.subheader("Select Classifiers")
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

selected_classifiers = []
for name, clf in classifiers.items():
    if st.sidebar.checkbox(name, value=True):
        selected_classifiers.append((name, clf))

# Voting type
voting_type = st.sidebar.radio("Voting Type", ["hard", "soft"])

# Run button
if st.sidebar.button("🚀 Run Voting Classifier") and selected_classifiers:
    
    # Get selected dataset
    X, y = datasets[dataset_name]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=selected_classifiers,
        voting=voting_type
    )
    
    # Train all classifiers
    results = {}
    
    # Train individual classifiers
    for name, clf in selected_classifiers:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'classifier': clf,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Train voting classifier
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_voting = voting_clf.predict(X_test_scaled)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    results['Voting'] = {
        'classifier': voting_clf,
        'accuracy': accuracy_voting,
        'predictions': y_pred_voting
    }
    
    # Display results
    st.subheader("📊 Results")
    
    # Accuracy comparison
    st.write("### Accuracy Scores")
    accuracy_data = []
    for name, result in results.items():
        accuracy_data.append({
            'Classifier': name,
            'Accuracy': result['accuracy']
        })
    
    df_accuracy = pd.DataFrame(accuracy_data).sort_values('Accuracy', ascending=False)
    st.dataframe(df_accuracy.style.format({'Accuracy': '{:.3f}'}).highlight_max(subset=['Accuracy']))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    bars = ax.barh(names, accuracies, color=['skyblue' if name != 'Voting' else 'orange' for name in names])
    ax.set_xlabel('Accuracy')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}', 
                ha='left', va='center')
    
    st.pyplot(fig)
    
    # Dataset visualization
    st.write("### Dataset Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax.set_title(f"{dataset_name} Dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)
    
    # Decision boundaries
    st.write("### Decision Boundaries")
    
    # Create subplots
    n_classifiers = len(results)
    fig, axes = plt.subplots(2, (n_classifiers + 1) // 2, figsize=(15, 10))
    if n_classifiers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot decision boundaries
    for idx, (name, result) in enumerate(results.items()):
        if idx < len(axes):
            # Create mesh grid
            h = 0.02
            x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
            y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict
            Z = result['classifier'].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            axes[idx].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            axes[idx].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, 
                             cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
    
    # Hide empty subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed reports
    st.write("### Detailed Classification Reports")
    
    for name, result in results.items():
        with st.expander(f"📋 {name} - Classification Report"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Report:**")
                report = classification_report(y_test, result['predictions'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            
            with col2:
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, result['predictions'])
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

else:
    if not selected_classifiers:
        st.warning("⚠️ Please select at least one classifier from the sidebar!")
    st.info("👈 Configure your settings in the sidebar and click 'Run Voting Classifier'")

st.sidebar.markdown("---")
st.sidebar.info("**Instructions:**\n1. Select a dataset\n2. Choose classifiers\n3. Select voting type\n4. Click Run!")