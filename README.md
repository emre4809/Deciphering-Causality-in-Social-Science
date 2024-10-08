# Deciphering-Causality-in-Social-Science
1. Data Preparation:
• Convert PDF documents to text.
• Perform post-processing to clean and format the text.
• Segment the text into individual sentences.
• Store the data in an appropriate format like CSV.
2. Model Application and Comparative Analysis
• Use our fine-tuned ssc-bert model on Hugging Face to classify sentences as causal or non-causal.
• Use packages like LIME for interpretation and comparative analysis of our fine tuned ssc-bert and general unicausal-seq-baseline model performance in classifying causal from non-causal sentences.
• Retain and store the extracted causal sentences in the appropriate format, such as CSV, for further analysis.
3. Annotation and Platform Preparation (Collaboration between all students):
• Prepare the labeling platform (Prodigy or Doccano) by writing guidelines and providing necessary information as directed by the supervisors.
• Define roles for labelers and ensure the platform is ready for annotation.
• Annotate causal sentences’ relations and entities based on the defined guidelines by supervisors.
• Compute inter-agreement scores to ensure consistency and reliability in the annotations.
4. Exploratory Text Mining:
• Analyze causal relationship patterns in the annotated dataset.
• Examine the impact of entities on these relations.
• Investigate co-occurrence patterns and perform correlation analysis.
• Visualize causal networks.
