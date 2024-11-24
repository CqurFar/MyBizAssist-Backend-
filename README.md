# **MyBizAssist-Backend-**  
<p style="text-align: center;">This repository contains the backend part of the project. It is currently in the alpha version, so functionality and performance may change significantly as development progresses.</p>

### **Tasks Overview**

#### **1. Website Parsing and DataFrame Creation**
The first stage of the project involved parsing multiple websites to gather relevant data for analysis.  
- A script was created to **parse websites** and extract useful information such as initiatives, news, and updates. For each website, a **separate DataFrame** was generated to organize the extracted data.
- To keep the data up-to-date, an **automated periodic update** system was implemented. This system runs every *n (24)* hours, ensuring that new initiatives or changes are captured promptly.
- Updates are performed by checking for changes in the **HTML elements** and **links** on the websites, allowing us to identify and retrieve newly published content automatically.

#### **2. Data Filtering and Text Preprocessing**
Once the data was collected, it underwent preprocessing to prepare it for further analysis.  
- The raw data in the form of DataFrames was **filtered** based on predefined criteria, such as removing irrelevant information or entries with missing values.
- Advanced **text preprocessing** techniques were applied to the content, including:
  - **Tokenization**: Breaking down text into smaller, manageable units such as words or phrases.
  - **Lemmatization**: Reducing words to their base or root form, enhancing the quality of the data for subsequent analysis.

#### **3. Data Unification and Splitting**
With data collected from different sources, the next step was to merge and prepare it for model training.  
- All the individual DataFrames from various sources were **unified** into a single standardized format. This involved ensuring consistency across data types, column names, and structures.
- After unification, the data was **split** into **train** and **test** datasets, facilitating the creation and evaluation of machine learning models.

#### **4. Text Classification with Machine Learning**
The next task was to classify the text data using machine learning algorithms.  
- A **CatBoost** model was trained using labeled data from trusted sources, such as government websites or other relevant references. This model was specifically designed for **classification tasks**, allowing us to categorize the text into meaningful labels.
- Once trained, the **CatBoost model** was used to classify texts from other sources, providing insights into their content and helping to organize them based on predefined categories.

#### **5. Topic Modeling with LDA**  
In this stage, we aimed to identify underlying themes within the text data using topic modeling techniques.  
- All the collected DataFrames were **merged** into a single dataset for comprehensive topic analysis.
- **Latent Dirichlet Allocation (LDA)** was applied to uncover hidden topics within the texts. LDA is an unsupervised machine learning algorithm that can discover topics by analyzing the co-occurrence of words across documents.
- Additionally, a **dictionary-based clustering** approach was explored. By creating a predefined set of keywords or concepts, we were able to group the texts into clusters based on shared vocabulary, providing an alternative method for topic discovery.

---

### **Technologies (models)**  
- **nlp**: used for advanced text processing and analysis  
- **CatBoost**: employed for classification tasks
- **LDA**: used for uncovering latent topics within text data
