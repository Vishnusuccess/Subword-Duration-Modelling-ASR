import os
import json
import pandas as pd
from glob import glob
import numpy as np

# Function to load the JSON data
def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read().strip()

        if not data:
            raise ValueError("Error: JSON file is empty!")

        return json.loads(data)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except ValueError as ve:
        print(ve)
        return None

# Function to extract the relevant fields
def extract_relevant_data(json_data, folder_name):
    extracted_data = []

    # Extract text details from the JSON
    sentence = json_data.get('result', {}).get('text', np.nan)
    normalized_sentence = json_data.get('result', {}).get('text_normalized', np.nan)

    # Extract audio info
    audio_info = json_data.get('audio_info', {})
    clipping = audio_info.get('clipping', np.nan)
    dynamic_ratio = audio_info.get('dynamic-ratio', np.nan)
    noise = audio_info.get('noise', np.nan)
    duration = audio_info.get('duration', np.nan)
    snr = audio_info.get('snr', np.nan)
    zcr = audio_info.get('zcr', np.nan)
    speech_energy = audio_info.get('speech_energy', np.nan)
    noise_energy = audio_info.get('noise_energy', np.nan)

    # Extract ASR model info
    asr_info = json_data.get('result', {}).get('asr_info', {})
    app_version = asr_info.get('app_version', np.nan)
    asr_model = asr_info.get('model', {})
    model_name = asr_model.get('name', np.nan)
    model_version = asr_model.get('version', np.nan)
    model_device = asr_model.get('device', np.nan)
    transcription_time = asr_info.get('transcription_time', np.nan)

    # Loop through the segments (in the "result" section)
    for segment in json_data.get('result', {}).get('segments', []):
        segment_text = segment.get('text', np.nan)
        segment_normalized_text = segment.get('text_normalized', np.nan)
        
        # Loop through the words within the segment
        for word_info in segment.get('words', []):
            word = word_info.get('word', np.nan)
            normalized_word = word_info.get('word_normalized', np.nan)
            word_start = word_info.get('start', np.nan)
            word_duration = word_info.get('duration', np.nan)

            # Loop through the phones (phonemes) for the word
            for phone_info in word_info.get('phones', []):
                phone = phone_info.get('phone', np.nan)
                phone_start = phone_info.get('start', np.nan)
                phone_duration = phone_info.get('duration', np.nan)
                phone_class = phone_info.get('class', np.nan)

                extracted_data.append({
                    'folder_name': folder_name, 
                    'sentence': sentence,
                    'normalized_sentence': normalized_sentence,
                    'segment_text': segment_text,
                    'segment_normalized_text': segment_normalized_text,
                    'word': word,
                    'normalized_word': normalized_word,
                    'word_start': word_start,
                    'word_duration': word_duration,
                    'phone': phone,
                    'phone_start': phone_start,
                    'phone_duration': phone_duration,
                    'phone_class': phone_class,
                    'clipping': clipping,
                    'dynamic_ratio': dynamic_ratio,
                    'noise': noise,
                    'audio_duration': duration,
                    'snr': snr,
                    'zcr': zcr,
                    'speech_energy': speech_energy,
                    'noise_energy': noise_energy,
                    'app_version': app_version,
                    'model_name': model_name,
                    'model_version': model_version,
                    'model_device': model_device,
                    'transcription_time': transcription_time,
                })
    
    return extracted_data

# Function to standardize JSON structure by adding missing fields or default values
def standardize_json(json_data):

    # Confirm the root contains the 'result' field
    if not isinstance(json_data, dict):
        return None

    if 'result' not in json_data:
        json_data['result'] = {}

    result = json_data['result']

    # Check that 'text' and 'text_normalized' exist
    result['text'] = result.get('text', "")
    result['text_normalized'] = result.get('text_normalized', "")

    if 'segments' not in result:
        result['segments'] = []

    # Check that every segment has the required fields
    for segment in result['segments']:
        segment['text'] = segment.get('text', "")
        segment['text_normalized'] = segment.get('text_normalized', "")
        if 'words' not in segment:
            segment['words'] = []

        for word_info in segment['words']:
            word_info['word'] = word_info.get('word', "")
            word_info['word_normalized'] = word_info.get('word_normalized', "")
            word_info['start'] = word_info.get('start', 0)
            word_info['duration'] = word_info.get('duration', 0)

            if 'phones' not in word_info:
                word_info['phones'] = []

            for phone_info in word_info['phones']:
                phone_info['phone'] = phone_info.get('phone', "")
                phone_info['start'] = phone_info.get('start', 0)
                phone_info['duration'] = phone_info.get('duration', 0)
                phone_info['class'] = phone_info.get('class', "")

    return json_data

def inspect_json_structure(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        json_files = glob(os.path.join(folder_path, "*.json"))
        
        # Inspect the JSON files in each folder
        for json_file in json_files[:5]: 
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = f.read().strip()

                if not data:
                    print(f"Empty file found: {json_file}")
                    continue

                json_data = json.loads(data)
                
                print(f"Inspecting file: {json_file}")
                print(f"Keys at root level: {json_data.keys()}")
                print(f"Result section: {json_data.get('result', {}).keys()}")
                print("-----")
                
            except Exception as e:
                print(f"Error reading file {json_file}: {e}")

# Main function to process all folders
def process_folders(directory):
    all_data = []  
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    total_extracted = 0
    total_discrepancies = 0
    
    print(f"Total number of folders: {len(folders)}")
    
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        json_files = glob(os.path.join(folder_path, "*.json"))
        
        for json_file in json_files:
            try:
                json_data = load_json(json_file)
                if json_data:
                    standardized_json = standardize_json(json_data)
                    if standardized_json:
                        extracted_data = extract_relevant_data(standardized_json, folder)
                        all_data.extend(extracted_data)
                        total_extracted += len(extracted_data)
                    else:
                        print(f"Skipping invalid structure in {json_file}.")
                        total_discrepancies += 1
                else:
                    print(f"Skipping corrupted file {json_file}.")
                    total_discrepancies += 1
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                total_discrepancies += 1

    # Save extracted data to CSV if any
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("extracted_data_all_folders.csv", index=False)
        print(df.head())
        print(df.columns)
        print("Extracted data saved to extracted_data_all_folders.csv")
    else:
        print("No valid data to save.")
    
    # Print summary of the extraction process
    print(f"Total records extracted: {total_extracted}")
    print(f"Total discrepancies (corrupted/removed files): {total_discrepancies}")


directory_path = "/Users/vishnu/Downloads/american_english"

# Inspect the JSON structure to understand inconsistencies
inspect_json_structure(directory_path)

# Process the folders and standardize the data
process_folders(directory_path)
