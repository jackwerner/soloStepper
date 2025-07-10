import librosa
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd

class GuitarSoloAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.y, self.sr = librosa.load(audio_file)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
    def detect_phrases(self, min_phrase_length=2.0, max_phrase_gap=1.5):
        """
        Detect guitar phrases based on onset density and spectral features
        """
        # Detect onsets (note beginnings)
        onset_frames = librosa.onset.onset_detect(
            y=self.y, 
            sr=self.sr, 
            units='time',
            hop_length=512,
            backtrack=True
        )
        
        # Get spectral features to identify guitar-like sounds
        spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        
        # Convert to time-based analysis
        time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=self.sr)
        
        # Identify active regions (where guitar is likely playing)
        # Guitar typically has higher spectral centroid during solos
        centroid_threshold = np.percentile(spectral_centroids, 70)
        active_regions = spectral_centroids > centroid_threshold
        
        # Group consecutive onsets into phrases
        phrases = []
        if len(onset_frames) > 0:
            phrase_start = onset_frames[0]
            last_onset = onset_frames[0]
            
            for onset in onset_frames[1:]:
                # If gap between onsets is too large, end current phrase
                if onset - last_onset > max_phrase_gap:
                    if last_onset - phrase_start >= min_phrase_length:
                        phrases.append({
                            'start': phrase_start,
                            'end': last_onset,
                            'duration': last_onset - phrase_start,
                            'onset_count': len([o for o in onset_frames if phrase_start <= o <= last_onset])
                        })
                    phrase_start = onset
                last_onset = onset
            
            # Add the last phrase
            if last_onset - phrase_start >= min_phrase_length:
                phrases.append({
                    'start': phrase_start,
                    'end': last_onset,
                    'duration': last_onset - phrase_start,
                    'onset_count': len([o for o in onset_frames if phrase_start <= o <= last_onset])
                })
        
        return phrases
    
    def detect_beats_and_measures(self):
        """
        Detect beats and estimate measure boundaries with automatic downbeat alignment
        """
        try:
            # Get tempo and beat locations
            tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, units='time')
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
            else:
                tempo = float(tempo)
            if len(beats) < 4:
                return tempo, beats, []

            # Estimate average beat interval
            beat_intervals = np.diff(beats)
            avg_beat_interval = np.median(beat_intervals)

            # Find the first downbeat BEFORE or at the first detected beat
            # (Assume the first bar starts at or before the first beat)
            first_beat = beats[0]
            # Find the time of the first downbeat (could be negative, but that's ok)
            first_downbeat = first_beat - (first_beat % (4 * avg_beat_interval))

            # Generate all downbeat times from the first downbeat up to the last beat
            downbeats = np.arange(first_downbeat, beats[-1] + 4 * avg_beat_interval, 4 * avg_beat_interval)

            # For each downbeat, find the closest 4 beats to define a measure
            measures = []
            for downbeat in downbeats:
                # Find the 4 beats that are closest to this measure
                measure_beats = []
                for i in range(4):
                    target_time = downbeat + i * avg_beat_interval
                    # Find the closest actual beat
                    idx = np.argmin(np.abs(beats - target_time))
                    measure_beats.append(beats[idx])
                # Only add if the measure is within the song
                if measure_beats[-1] <= beats[-1]:
                    measures.append({
                        'start': measure_beats[0],
                        'end': measure_beats[-1],
                        'beats': measure_beats
                    })

            print(f"Auto-aligned {len(measures)} measures starting from {measures[0]['start']:.2f}s")
            return tempo, beats, measures

        except Exception as e:
            print(f"Beat detection failed: {e}")
            return 120.0, [], []

    def find_downbeat_index(self, beats):
        """
        Find the index of the true downbeat (beat 1) in the detected beats array
        """
        
        # Method 1: Use onset strength to find the strongest beats
        onset_strength = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=self.sr)
        
        # Calculate strength at each beat position
        beat_strengths = []
        for beat_time in beats[:min(len(beats), 16)]:  # Check first 16 beats
            # Find the closest onset strength measurement to this beat
            closest_idx = np.argmin(np.abs(onset_times - beat_time))
            beat_strengths.append(onset_strength[closest_idx])
        
        # Method 2: Look for patterns in beat strength (downbeats are often strongest)
        # Try each possible downbeat position (0, 1, 2, 3) and see which gives the strongest pattern
        best_score = -1
        best_offset = 0
        
        for offset in range(4):
            if offset >= len(beat_strengths):
                continue
            
            # Calculate average strength for beats that would be downbeats with this offset
            downbeat_strengths = []
            for i in range(offset, len(beat_strengths), 4):
                downbeat_strengths.append(beat_strengths[i])
            
            if len(downbeat_strengths) >= 2:
                avg_downbeat_strength = np.mean(downbeat_strengths)
                
                # Also calculate strength of other beats
                other_beats = []
                for i in range(len(beat_strengths)):
                    if (i - offset) % 4 != 0:  # Not a downbeat with this offset
                        other_beats.append(beat_strengths[i])
                
                if len(other_beats) > 0:
                    avg_other_strength = np.mean(other_beats)
                    strength_ratio = avg_downbeat_strength / (avg_other_strength + 1e-10)
                    
                    # Higher ratio indicates this offset gives stronger downbeats
                    if strength_ratio > best_score:
                        best_score = strength_ratio
                        best_offset = offset
        
        return best_offset
    
    def detect_chords(self):
        """
        Detect chord progressions throughout the song
        """
        try:
            return self.detect_chords_alternative()
        except Exception as e:
            print(f"Chord detection failed: {e}")
            return []
    
    def detect_chords_alternative(self):
        """
        Alternative chord detection using librosa
        """
        # Get chromagram
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        
        # Simple chord detection based on chroma
        hop_length = 512
        frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), 
                                           sr=self.sr, hop_length=hop_length)
        
        chord_timeline = []
        for i, time in enumerate(frame_times):
            # Find dominant chord based on chroma
            dominant_chroma = np.argmax(chroma[:, i])
            chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                          'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            chord_timeline.append({
                'start': time,
                'chord': chord_names[dominant_chroma],
                'end': time + (hop_length / self.sr)
            })
        
        return chord_timeline
    
    def match_phrases_with_chords(self, phrases, chords):
        """
        Match detected phrases with their corresponding chords
        """
        phrase_chord_analysis = []
        
        for i, phrase in enumerate(phrases):
            # Find chords that overlap with this phrase
            overlapping_chords = []
            for chord in chords:
                chord_start = chord['start']
                chord_end = chord['end'] if chord['end'] else chord_start + 2.0  # Default 2s duration
                
                # Check if chord overlaps with phrase
                if (chord_start <= phrase['end'] and chord_end >= phrase['start']):
                    overlap_start = max(phrase['start'], chord_start)
                    overlap_end = min(phrase['end'], chord_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration > 0:
                        overlapping_chords.append({
                            'chord': chord['chord'],
                            'overlap_duration': overlap_duration,
                            'overlap_percentage': overlap_duration / phrase['duration'] * 100
                        })
            
            phrase_chord_analysis.append({
                'phrase_number': i + 1,
                'start_time': phrase['start'],
                'end_time': phrase['end'],
                'duration': phrase['duration'],
                'onset_count': phrase['onset_count'],
                'chords': overlapping_chords,
                'primary_chord': overlapping_chords[0]['chord'] if overlapping_chords else 'Unknown'
            })
        
        return phrase_chord_analysis
    
    def analyze_solo(self, start_time=None, end_time=None):
        """
        Complete analysis of guitar solo section
        """
        print(f"Analyzing audio file: {self.audio_file}")
        print(f"Duration: {self.duration:.2f} seconds")
        
        # If specific time range provided, crop audio
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            self.y = self.y[start_sample:end_sample]
            print(f"Analyzing section: {start_time:.2f}s - {end_time:.2f}s")
        
        # Detect phrases
        print("\nDetecting guitar phrases...")
        phrases = self.detect_phrases()
        print(f"Found {len(phrases)} phrases")
        
        # Detect beats and measures
        print("\nDetecting beats and measures...")
        tempo, beats, measures = self.detect_beats_and_measures()
        print(f"Tempo: {tempo:.1f} BPM")
        print(f"Found {len(measures)} measures")
        
        # Detect chords
        print("\nDetecting chords...")
        chords = self.detect_chords()
        print(f"Found {len(chords)} chord changes")
        
        # Match phrases with chords
        print("\nMatching phrases with chords...")
        analysis = self.match_phrases_with_chords(phrases, chords)
        
        return {
            'phrases': phrases,
            'beats': beats,
            'measures': measures,
            'chords': chords,
            'analysis': analysis,
            'tempo': tempo
        }
    
    def print_analysis(self, results):
        """
        Print formatted analysis results
        """
        print("\n" + "="*60)
        print("GUITAR SOLO PHRASE ANALYSIS")
        print("="*60)
        
        for phrase in results['analysis']:
            print(f"\nPhrase {phrase['phrase_number']}:")
            print(f"  Time: {phrase['start_time']:.2f}s - {phrase['end_time']:.2f}s")
            print(f"  Duration: {phrase['duration']:.2f}s")
            print(f"  Notes/Onsets: {phrase['onset_count']}")
            print(f"  Primary Chord: {phrase['primary_chord']}")
            
            if len(phrase['chords']) > 1:
                print("  All Chords:")
                for chord in phrase['chords']:
                    print(f"    {chord['chord']} ({chord['overlap_percentage']:.1f}% of phrase)")
    
    def save_analysis_to_csv(self, results, output_file="guitar_solo_analysis.csv"):
        """
        Save analysis results to CSV file
        """
        data = []
        for phrase in results['analysis']:
            chord_list = ", ".join([c['chord'] for c in phrase['chords']])
            data.append({
                'Phrase': phrase['phrase_number'],
                'Start_Time': phrase['start_time'],
                'End_Time': phrase['end_time'],
                'Duration': phrase['duration'],
                'Onset_Count': phrase['onset_count'],
                'Primary_Chord': phrase['primary_chord'],
                'All_Chords': chord_list
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"\nAnalysis saved to {output_file}")

    def detect_guitar_licks(self, min_lick_measures=1, max_lick_measures=2):
        """
        Detect guitar licks using measure-aligned boundaries
        Ensures all licks start and end on downbeats (beat 1 of measures)
        """
        # Get beats and measures first (no manual offset)
        tempo, beats, measures = self.detect_beats_and_measures()
        
        if len(measures) < 1:
            print("Not enough measures detected for lick analysis")
            return []
        
        print(f"Using {len(measures)} measures for lick detection")
        print(f"First measure: {measures[0]['start']:.2f}s - {measures[0]['end']:.2f}s")
        print(f"Beats in first measure: {[f'{b:.2f}' for b in measures[0]['beats']]}")
        
        # Get onset information for activity detection
        onset_frames = librosa.onset.onset_detect(
            y=self.y, 
            sr=self.sr, 
            units='time',
            hop_length=512,
            backtrack=True
        )
        
        licks = []
        
        # Create licks based on measure boundaries
        i = 0
        while i <= len(measures) - min_lick_measures:
            lick_found = False
            
            # Try different lick lengths in measures
            for lick_length in range(min_lick_measures, min(max_lick_measures + 1, len(measures) - i + 1)):
                start_time = measures[i]['start']  # Always start on downbeat
                
                # End time is the start of the next measure after this lick
                if i + lick_length < len(measures):
                    end_time = measures[i + lick_length]['start']
                else:
                    # If this is the last possible lick, end at the end of the last measure
                    end_time = measures[i + lick_length - 1]['end']
                
                # Count onsets in this region
                region_onsets = [o for o in onset_frames if start_time <= o <= end_time]
                
                # Use advanced guitar detection
                guitar_score = self.detect_lead_guitar_activity(start_time, end_time)
                
                # Only create lick if it has good guitar characteristics
                if len(region_onsets) >= 1 and guitar_score > 0.4:
                    # Collect all beat positions for this lick
                    beat_positions = []
                    for measure_idx in range(i, min(i + lick_length, len(measures))):
                        beat_positions.extend(measures[measure_idx]['beats'])
                    
                    licks.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time,
                        'beats': lick_length * 4,  # Each measure has 4 beats
                        'measures': lick_length,
                        'onset_count': len(region_onsets),
                        'beat_positions': beat_positions,
                        'activity_score': guitar_score,
                        'guitar_confidence': guitar_score,
                        'measure_numbers': list(range(i + 1, i + lick_length + 1))  # For debugging
                    })
                    
                    # Advance to the end of this lick to avoid overlaps
                    i += lick_length
                    lick_found = True
                    break
            
            # If no lick was found, advance by min_lick_measures to find the next potential lick
            if not lick_found:
                i += min_lick_measures
        
        # Filter by guitar confidence
        guitar_licks = [lick for lick in licks if lick['guitar_confidence'] > 0.5]
        
        return guitar_licks

    def detect_lead_guitar_activity(self, region_start, region_end):
        """
        Advanced guitar detection using multiple audio features to distinguish
        lead guitar from vocals, rhythm guitar, and other instruments
        """
        # Extract the audio segment for this region
        start_sample = int(region_start * self.sr)
        end_sample = int(region_end * self.sr)
        region_audio = self.y[start_sample:end_sample]
        
        if len(region_audio) < 1024:  # Too short to analyze
            return 0.0
        
        # 1. Harmonic-Percussive Separation
        harmonic, percussive = librosa.effects.hpss(region_audio)
        harmonic_power = np.mean(harmonic**2)
        percussive_power = np.mean(percussive**2)
        hp_ratio = harmonic_power / (percussive_power + 1e-10)  # Lead guitar is more harmonic
        
        # 2. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=region_audio, sr=self.sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=region_audio, sr=self.sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=region_audio, sr=self.sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(region_audio))
        
        # 3. Frequency analysis - focus on guitar range
        fft = np.abs(np.fft.fft(region_audio))
        freqs = np.fft.fftfreq(len(region_audio), 1/self.sr)
        
        # Guitar fundamental frequency range (roughly 80Hz - 1000Hz)
        guitar_range_mask = (freqs >= 80) & (freqs <= 1000)
        guitar_power = np.sum(fft[guitar_range_mask]**2)
        
        # High frequency content (guitar harmonics vs vocal formants)
        high_freq_mask = (freqs >= 1000) & (freqs <= 4000)
        high_freq_power = np.sum(fft[high_freq_mask]**2)
        
        # Very high frequency (guitar brightness)
        very_high_freq_mask = (freqs >= 4000) & (freqs <= 8000)
        very_high_freq_power = np.sum(fft[very_high_freq_mask]**2)
        
        total_power = np.sum(fft**2) + 1e-10
        guitar_ratio = guitar_power / total_power
        high_freq_ratio = high_freq_power / total_power
        brightness_ratio = very_high_freq_power / total_power
        
        # 4. Melodic variation (pitch changes indicate lead guitar)
        pitches, magnitudes = librosa.piptrack(y=region_audio, sr=self.sr, threshold=0.1)
        valid_pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        pitch_variation = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
        
        # 5. Onset density and rhythm complexity
        region_onsets = librosa.onset.onset_detect(y=region_audio, sr=self.sr, units='time')
        onset_density = len(region_onsets) / (region_end - region_start)
        
        # Calculate guitar score (0-1)
        guitar_score = 0.0
        
        # Harmonic vs percussive (lead guitar is harmonic)
        guitar_score += min(hp_ratio / 5.0, 0.2)  # Max 0.2 points
        
        # Spectral characteristics
        # Lead guitar typically has higher spectral centroid than rhythm guitar
        if 1000 < spectral_centroid < 3000:  # Sweet spot for lead guitar
            guitar_score += 0.15
        elif spectral_centroid > 3000:  # Too high (might be vocals)
            guitar_score -= 0.1
        
        # Bandwidth - lead guitar has wider bandwidth than vocals
        if spectral_bandwidth > 500:
            guitar_score += 0.1
        
        # Frequency content ratios
        guitar_score += min(guitar_ratio * 2, 0.2)  # Guitar fundamental range
        guitar_score += min(high_freq_ratio * 3, 0.15)  # Guitar harmonics
        guitar_score += min(brightness_ratio * 5, 0.1)  # Guitar brightness
        
        # Pitch variation (lead guitar changes pitch more than rhythm)
        if pitch_variation > 50:  # Significant pitch movement
            guitar_score += 0.15
        elif pitch_variation < 10:  # Too static (might be strumming or held vocal)
            guitar_score -= 0.1
        
        # Onset density (lead guitar has moderate complexity)
        if 2 < onset_density < 8:  # Good range for lead guitar
            guitar_score += 0.1
        elif onset_density > 10:  # Too busy (might be drums)
            guitar_score -= 0.1
        elif onset_density < 1:  # Too sparse
            guitar_score -= 0.05
        
        # Zero crossing rate (distinguish from vocals)
        if 0.05 < zero_crossing_rate < 0.3:  # Typical for guitar
            guitar_score += 0.05
        
        return max(0.0, min(1.0, guitar_score))  # Clamp between 0 and 1

    def detect_chords_improved(self):
        """
        Improved chord detection with better temporal resolution
        and more sophisticated chord recognition
        """
        # Get chromagram with higher time resolution
        hop_length = 256  # Smaller hop for better temporal resolution
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=hop_length)
        
        # Smooth the chromagram to reduce noise
        from scipy.ndimage import uniform_filter1d
        chroma_smooth = uniform_filter1d(chroma, size=5, axis=1)
        
        frame_times = librosa.frames_to_time(np.arange(chroma_smooth.shape[1]), 
                                           sr=self.sr, hop_length=hop_length)
        
        # Define more comprehensive chord templates for blues/rock
        chord_templates = {
            # Major chords
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],    # C major
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],    # D major
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],    # E major
            'F': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],    # F major
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],    # G major
            'A': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],    # A major
            'B': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],    # B major
            'Bb': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],   # Bb major
            
            # Minor chords
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],   # C minor
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],   # D minor
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],   # E minor
            'Fm': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],   # F minor
            'Gm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],   # G minor
            'Am': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],   # A minor
            'Bm': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],   # B minor
            
            # 7th chords (common in blues)
            'C7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],   # C7
            'D7': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],   # D7
            'E7': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],   # E7
            'G7': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],   # G7
            'A7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],   # A7
        }
        
        chord_timeline = []
        window_size = 8  # Larger window for more stable chord detection
        
        for i in range(0, chroma_smooth.shape[1] - window_size, window_size//2):  # 50% overlap
            # Average chroma over window
            window_chroma = np.mean(chroma_smooth[:, i:i+window_size], axis=1)
            window_time = frame_times[i + window_size//2]
            
            # Normalize chroma
            if np.sum(window_chroma) > 0:
                window_chroma = window_chroma / np.sum(window_chroma)
            
            # Find best matching chord
            best_chord = 'Unknown'
            best_score = -1
            
            for chord_name, template in chord_templates.items():
                template_normalized = np.array(template) / np.sum(template) if np.sum(template) > 0 else np.array(template)
                # Calculate correlation with template
                score = np.corrcoef(window_chroma, template_normalized)[0, 1]
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_chord = chord_name
            
            # Lower threshold and add chord if correlation is decent
            if best_score > 0.2:  # Lower threshold
                chord_timeline.append({
                    'start': window_time - (window_size * hop_length / self.sr / 2),
                    'end': window_time + (window_size * hop_length / self.sr / 2),
                    'chord': best_chord,
                    'confidence': best_score
                })
        
        return chord_timeline

    def match_licks_with_chords(self, licks, chords):
        """
        Match detected licks with their corresponding chords
        """
        lick_chord_analysis = []
        
        for i, lick in enumerate(licks):
            # Find chords that overlap with this lick
            overlapping_chords = []
            chord_counts = {}
            
            for chord in chords:
                # Check if chord overlaps with lick
                if (chord['start'] <= lick['end'] and chord['end'] >= lick['start']):
                    overlap_start = max(lick['start'], chord['start'])
                    overlap_end = min(lick['end'], chord['end'])
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration > 0:
                        overlapping_chords.append({
                            'chord': chord['chord'],
                            'overlap_duration': overlap_duration,
                            'overlap_percentage': overlap_duration / lick['duration'] * 100,
                            'confidence': chord.get('confidence', 0.5)
                        })
                        
                        # Count chord occurrences
                        chord_name = chord['chord']
                        if chord_name not in chord_counts:
                            chord_counts[chord_name] = 0
                        chord_counts[chord_name] += overlap_duration
            
            # Determine primary chord (most prevalent by duration)
            primary_chord = 'Unknown'
            if chord_counts:
                primary_chord = max(chord_counts.keys(), key=lambda k: chord_counts[k])
            
            lick_chord_analysis.append({
                'lick_number': i + 1,
                'start_time': lick['start'],
                'end_time': lick['end'],
                'duration': lick['duration'],
                'beats': lick['beats'],
                'onset_count': lick['onset_count'],
                'activity_score': lick['activity_score'],
                'chords': overlapping_chords,
                'primary_chord': primary_chord,
                'beat_positions': lick['beat_positions']
            })
        
        return lick_chord_analysis

    def analyze_guitar_licks(self, start_time=None, end_time=None, min_lick_measures=1, max_lick_measures=2):
        """
        Complete analysis focused on measure-aligned guitar licks and their underlying chords
        """
        print(f"Analyzing audio file: {self.audio_file}")
        print(f"Duration: {self.duration:.2f} seconds")
        
        # If specific time range provided, crop audio
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            self.y = self.y[start_sample:end_sample]
            print(f"Analyzing section: {start_time:.2f}s - {end_time:.2f}s")
        
        # Detect measure-aligned guitar licks (no manual offset)
        print("\nDetecting measure-aligned guitar licks...")
        licks = self.detect_guitar_licks(
            min_lick_measures=min_lick_measures, 
            max_lick_measures=max_lick_measures
        )
        print(f"Found {len(licks)} guitar licks")
        
        # Detect beats and measures
        print("\nDetecting beats and measures...")
        tempo, beats, measures = self.detect_beats_and_measures()
        print(f"Tempo: {tempo:.1f} BPM")
        print(f"Found {len(measures)} measures")
        
        # Detect chords with improved method
        print("\nDetecting chords...")
        chords = self.detect_chords_improved()
        print(f"Found {len(chords)} chord regions")
        
        # Match licks with chords
        print("\nMatching licks with chords...")
        analysis = self.match_licks_with_chords(licks, chords)
        
        return {
            'licks': licks,
            'beats': beats,
            'measures': measures,
            'chords': chords,
            'analysis': analysis,
            'tempo': tempo
        }

    def print_lick_analysis(self, results):
        """
        Print formatted lick analysis results
        """
        print("\n" + "="*60)
        print("GUITAR LICK ANALYSIS (MEASURE-ALIGNED)")
        print("="*60)
        
        tempo = results.get('tempo', 120)
        
        for lick in results['analysis']:
            measures = lick.get('measures', lick['beats'] // 4)
            print(f"\nLick {lick['lick_number']}:")
            print(f"  Time: {lick['start_time']:.2f}s - {lick['end_time']:.2f}s")
            print(f"  Duration: {lick['duration']:.2f}s ({measures} measures, {lick['beats']} beats)")
            print(f"  Primary Chord: {lick['primary_chord']}")
            print(f"  Activity Score: {lick['activity_score']:.2f}")
            print(f"  Notes/Onsets: {lick['onset_count']}")
            
            if len(lick['chords']) > 1:
                print("  All Chords:")
                for chord in lick['chords']:
                    print(f"    {chord['chord']} ({chord['overlap_percentage']:.1f}% of lick)")

    def save_lick_analysis_to_csv(self, results, output_file="guitar_lick_analysis.csv"):
        """
        Save lick analysis results to CSV file
        """
        data = []
        for lick in results['analysis']:
            chord_list = ", ".join([c['chord'] for c in lick['chords']])
            data.append({
                'Lick': lick['lick_number'],
                'Start_Time': lick['start_time'],
                'End_Time': lick['end_time'],
                'Duration': lick['duration'],
                'Beats': lick['beats'],
                'Onset_Count': lick['onset_count'],
                'Activity_Score': lick['activity_score'],
                'Primary_Chord': lick['primary_chord'],
                'All_Chords': chord_list
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"\nLick analysis saved to {output_file}")

    def export_for_web_dashboard(self, results, audio_filename, output_json="dashboard_data.json"):
        """
        Export analysis results in JSON format for the web dashboard
        """
        import json
        import os
        
        # Prepare data for web interface
        dashboard_data = {
            'audio_file': os.path.basename(audio_filename),
            'tempo': results.get('tempo', 120),
            'total_duration': self.duration,
            'licks': []
        }
        
        for lick in results['analysis']:
            lick_data = {
                'id': lick['lick_number'],
                'start_time': round(lick['start_time'], 2),
                'end_time': round(lick['end_time'], 2),
                'duration': round(lick['duration'], 2),
                'beats': lick['beats'],
                'onset_count': lick['onset_count'],
                'activity_score': round(lick['activity_score'], 2),
                'primary_chord': lick['primary_chord'],
                'all_chords': [c['chord'] for c in lick['chords']],
                'beat_positions': [round(pos, 2) for pos in lick['beat_positions']]
            }
            dashboard_data['licks'].append(lick_data)
        
        # Save JSON file
        with open(output_json, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"\nDashboard data exported to {output_json}")
        return dashboard_data

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GuitarSoloAnalyzer("Blue Sky.mp3")
    
    # Analyze guitar licks - focus on the solo section with measure-aligned licks
    # Try manual_beat_offset=0, 1, 2, or 3 if automatic detection doesn't work well
    results = analyzer.analyze_guitar_licks(
        start_time=120, 
        end_time=300,
        min_lick_measures=1,  # Minimum 1 measure (4 beats)
        max_lick_measures=2,   # Maximum 2 measures (8 beats)
    )
    
    # Print results
    analyzer.print_lick_analysis(results)
    
    # Save to CSV with the new method
    analyzer.save_lick_analysis_to_csv(results, "texas_flood_licks.csv")
    
    # Export for web dashboard
    analyzer.export_for_web_dashboard(results, analyzer.audio_file, "dashboard_data.json")
    
    # Optional: Create a simple visualization
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(analyzer.y) / analyzer.sr, len(analyzer.y))
    plt.plot(time_axis, analyzer.y, alpha=0.6)
    plt.title("Audio Waveform")
    plt.ylabel("Amplitude")
    
    # Plot licks (changed from phrases)
    plt.subplot(3, 1, 2)
    for i, lick in enumerate(results['licks']):  # Changed from results['phrases']
        plt.barh(0, lick['duration'], left=lick['start'], height=0.5, 
                alpha=0.7, label=f"Lick {i+1}")
    plt.title("Detected Guitar Licks")  # Updated title
    plt.ylabel("Licks")  # Updated label
    
    # Plot beats
    plt.subplot(3, 1, 3)
    plt.vlines(results['beats'], 0, 1, colors='red', alpha=0.8, linewidth=1)
    plt.title("Beat Locations")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Beats")
    
    plt.tight_layout()
    plt.show()