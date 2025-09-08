
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load cleaned groundwater data
try:
    df = pd.read_csv("cleaned_groundwater_data.csv")
except FileNotFoundError:
    print("Error: cleaned_groundwater_data.csv not found. Please run excel_parser.py first.")
    exit()

# Prepare data for vector store
df['combined_text'] = df['STATE'].fillna('') + " " + \
                         df['DISTRICT'].fillna('') + " " + \
                         df['ASSESSMENT UNIT'].fillna('')
corpus_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=False)

def process_query_nlp(query):
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract specific locations (GPE - Geo-Political Entity)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    
    # Removed year extraction as 'ASSESSMENT YEAR' column is not present in the cleaned data
    # year = None 
    # for ent in doc.ents:
    #     if ent.label_ == "DATE" and ent.text.isdigit() and len(ent.text) == 4:
    #         year = int(ent.text)
    #         break
    # if year is None:
    #     for token in doc:
    #         if token.is_digit and len(token.text) == 4:
    #             year = int(token.text)
    #             break

    return entities, keywords, locations # Removed year from return

def search_groundwater_data(query_embedding, top_k=10):
    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for i in top_k_indices:
        results.append({
            "score": similarities[i],
            "data": df.iloc[i].to_dict()
        })
    return results

def generate_response(query):
    entities, keywords, locations = process_query_nlp(query) # Removed year from unpacking
    query_embedding = model.encode(query)
    search_results = search_groundwater_data(query_embedding, top_k=10) # Get top 10 relevant entries

    if not search_results:
        return "I could not find any relevant information for your query."

    # Filter results by identified location
    filtered_results = []
    target_state = None
    target_district = None

    for loc in locations:
        # Try to match a state or district from the identified locations
        # Using .str.contains with case=False for case-insensitive matching
        state_match = df[df['STATE'].fillna('').str.contains(loc, case=False, na=False)]
        district_match = df[df['DISTRICT'].fillna('').str.contains(loc, case=False, na=False)]

        if not state_match.empty:
            target_state = state_match['STATE'].iloc[0]
            break
        if not district_match.empty:
            target_district = district_match['DISTRICT'].iloc[0]
            # If district is found, also try to get its state for broader context
            if target_state is None:
                state_from_district = df[df['DISTRICT'].fillna('').str.contains(loc, case=False, na=False)]['STATE'].iloc[0]
                target_state = state_from_district
            break
    
    # If a specific location is identified, filter the search results
    if target_state:
        # Ensure we are filtering based on the *identified* target_state
        # Filter directly from the main DataFrame to ensure all matching rows are considered for aggregation
        filtered_df_by_location = df[df['STATE'].fillna('').str.contains(target_state, case=False, na=False)].copy()
        if target_district and target_district != 'nan': 
            filtered_df_by_location = filtered_df_by_location[filtered_df_by_location['DISTRICT'].fillna('').str.contains(target_district, case=False, na=False)].copy()
        
        # Convert filtered_df_by_location to a list of dicts to match the structure of search_results
        filtered_results = [{'score': 1.0, 'data': row.to_dict()} for index, row in filtered_df_by_location.iterrows()]

    else:
        # Fallback to the state/district of the top search result if no explicit location was identified
        # but still prefer filtering by the best match if it makes sense
        top_result_state = search_results[0]['data'].get('STATE')
        top_result_district = search_results[0]['data'].get('DISTRICT')
        
        if top_result_state:
            # Filter directly from the main DataFrame to ensure all matching rows are considered for aggregation
            filtered_df_by_location = df[df['STATE'].fillna('').str.contains(top_result_state, case=False, na=False)].copy()
            if top_result_district and 'groundwater availability' not in query.lower() and 'estimation' not in query.lower():
                filtered_df_by_location = filtered_df_by_location[filtered_df_by_location['DISTRICT'].fillna('').str.contains(top_result_district, case=False, na=False)].copy()
            
            filtered_results = [{'score': 1.0, 'data': row.to_dict()} for index, row in filtered_df_by_location.iterrows()]
        else:
            filtered_results = search_results # If still no strong location, use all top_k results
    
    if not filtered_results:
        return "I found some data, but I couldn't filter it by a specific location from your query. Please be more specific about the State or District."

    response_parts = []
    
    # Construct an overall summary for the identified location
    location_display = []
    if target_state:
        location_display.append(target_state)
    elif top_result_state: # Use top_result_state as a fallback for display if target_state was not directly identified
        location_display.append(top_result_state)

    if target_district and target_district != 'nan':
        location_display.append(f"District {target_district}")
    elif top_result_district and top_result_district != 'nan' and not target_district and ('groundwater availability' not in query.lower() and 'estimation' not in query.lower()):
        location_display.append(f"District {top_result_district}")

    if location_display:
        location_info = f"For {', '.join(location_display)}:\n"
    else:
        location_info = f"For the most relevant area:\n" # Fallback if no location identified

    response_parts.append(location_info)

    # Removed year filter logic as 'ASSESSMENT YEAR' column is not present in the cleaned data
    # if year:
    #     filtered_df_by_year = pd.DataFrame([res['data'] for res in filtered_results])
    #     filtered_df_by_year['ASSESSMENT YEAR'] = pd.to_numeric(filtered_df_by_year['ASSESSMENT YEAR'], errors='coerce')
    #     filtered_df_by_year = filtered_df_by_year[filtered_df_by_year['ASSESSMENT YEAR'] == year].copy()
    #     
    #     if filtered_df_by_year.empty:
    #         response_parts.append(f"No data found for the year {year} in this region.")
    #         return "\n".join(response_parts)
    #     
    #     filtered_results = [{'score': 1.0, 'data': row.to_dict()} for index, row in filtered_df_by_year.iterrows()]


    # Check for specific intents and gather data
    if "groundwater availability" in query.lower() or "estimation" in query.lower() or "overall" in query.lower():
        response_parts.append("Here's an overview of groundwater data:")
        
        # Aggregate data from the filtered_results which are now comprehensive for the location and year
        availabilities_fresh = []
        availabilities_saline = []
        rainfalls = []
        stages = []

        for res in filtered_results:
            data = res['data']
            
            # Ensure the values are not None or 'N/A' before appending
            fresh = data.get('Total Ground Water Availability in the area (ham) - Other Parameters Present - Fresh')
            if pd.notna(fresh):
                availabilities_fresh.append(fresh)

            saline = data.get('Total Ground Water Availability in the area (ham) - Other Parameters Present - Saline')
            if pd.notna(saline):
                availabilities_saline.append(saline)
            
            rainfall = data.get('Rainfall (mm) - Total')
            if pd.notna(rainfall):
                rainfalls.append(rainfall)

            stage = data.get('Stage of Ground Water Extraction (%)')
            if pd.notna(stage):
                stages.append(stage)
        
        if availabilities_fresh:
            response_parts.append(f"  - Average Fresh Groundwater Availability: {pd.Series(availabilities_fresh).astype(float).mean():.2f} ham")
        if availabilities_saline:
            response_parts.append(f"  - Average Saline Groundwater Availability: {pd.Series(availabilities_saline).astype(float).mean():.2f} ham")
        if rainfalls:
            response_parts.append(f"  - Average Total Rainfall: {pd.Series(rainfalls).astype(float).mean():.2f} mm")
        if stages:
            response_parts.append(f"  - Average Stage of Groundwater Extraction: {pd.Series(stages).astype(float).mean():.2f}%")
        
        if not (availabilities_fresh or availabilities_saline or rainfalls or stages):
            response_parts.append("  No specific numerical data found for these metrics in the identified region.")

    elif "rainfall data" in query.lower():
        response_parts.append("Here's the rainfall data:")
        # Iterate through filtered_results to present individual rainfall data if available
        for res in filtered_results:
            data = res['data']
            unit = data.get('ASSESSMENT UNIT', 'N/A')
            total_rainfall = data.get('Rainfall (mm) - Total', 'N/A')
            if pd.notna(total_rainfall) and unit != 'nan':
                response_parts.append(f"  - {unit}: {total_rainfall} mm")
            elif pd.notna(total_rainfall):
                response_parts.append(f"  - Total Rainfall: {total_rainfall} mm (for the area)")
        if not any(pd.notna(res['data'].get('Rainfall (mm) - Total')) for res in filtered_results):
            response_parts.append("  No rainfall data found for the identified region.")

    elif "stage of groundwater extraction" in query.lower():
        response_parts.append("Here's the stage of groundwater extraction data:")
        # Iterate through filtered_results to present individual stage data if available
        for res in filtered_results:
            data = res['data']
            unit = data.get('ASSESSMENT UNIT', 'N/A')
            stage_extraction = data.get('Stage of Ground Water Extraction (%)', 'N/A')
            if pd.notna(stage_extraction) and unit != 'nan':
                response_parts.append(f"  - {unit}: {stage_extraction}%")
            elif pd.notna(stage_extraction):
                response_parts.append(f"  - Stage of Groundwater Extraction: {stage_extraction}% (for the area)")
        if not any(pd.notna(res['data'].get('Stage of Ground Water Extraction (%)')) for res in filtered_results):
            response_parts.append("  No stage of groundwater extraction data found for the identified region.")

    else:
        # Default response if specific intent is not recognized but location is.
        response_parts.append("  I found general information for this area. Please specify what data you are looking for (e.g., 'groundwater availability', 'rainfall data', 'stage of extraction').")
    
    return "\n".join(response_parts)

if __name__ == "__main__":
    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        bot_response = generate_response(user_query)
        print(f"Bot: {bot_response}")
