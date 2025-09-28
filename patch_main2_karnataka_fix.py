#!/usr/bin/env python3
"""
Patch to fix Karnataka search issue in main2.py
"""

def create_patch():
    """Create a patch file to fix the Karnataka search issue"""
    
    patch_content = '''
# Patch for main2.py - Fix Karnataka search issue
# Replace lines 2595-2603 with the following:

    # Use advanced RAG pipeline with hybrid search, reranking, and query expansion
    print("🚀 Using Advanced RAG Pipeline...")
    candidate_results = advanced_search_with_rag(translated_query, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    
    # If no results found with location filters, try with broader search but keep location filters
    if not candidate_results and target_state:
        print("⚠️ No results with strict location filters, trying broader search...")
        # Try with broader district matching
        candidate_results = advanced_search_with_rag(translated_query, year=year, target_state=target_state, target_district=None, extracted_parameters=extracted_parameters)
    
    # If still no results and we have a specific state, try with just the state
    if not candidate_results and target_state:
        print("⚠️ No results with district filters, trying state-only search...")
        candidate_results = advanced_search_with_rag(translated_query, year=year, target_state=target_state, target_district=None, extracted_parameters=extracted_parameters)
    
    # Only if we have no state specified, try without location filters
    if not candidate_results and not target_state:
        print("⚠️ No results found, trying without location filters...")
        candidate_results = advanced_search_with_rag(translated_query, year=year, target_state=None, target_district=None, extracted_parameters=extracted_parameters)
    
    # If still no results, try with original query (before translation)
    if not candidate_results:
        print("⚠️ No results with translated query, trying original...")
        candidate_results = advanced_search_with_rag(original_query, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    
    # If still no results, try with common groundwater keywords but keep state filter
    if not candidate_results:
        print("⚠️ No results found, trying fallback search...")
        groundwater_query = "groundwater estimation data analysis"
        candidate_results = search_excel_chunks(groundwater_query, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
'''
    
    with open('karnataka_fix_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("Patch created: karnataka_fix_patch.txt")
    print("This patch fixes the Karnataka search issue by:")
    print("1. Keeping state filters when searching")
    print("2. Only removing filters if no state was specified")
    print("3. Ensuring Karnataka queries always return Karnataka data")

if __name__ == "__main__":
    create_patch()
