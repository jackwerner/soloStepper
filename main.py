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
        Robust beat and measure detection that handles 3/4 and 4/4 time signatures
        and can adapt to irregular timing and pauses
        """
        try:
            # Get basic tempo and beat tracking with only supported parameters
            tempo, beats = librosa.beat.beat_track(
                y=self.y, 
                sr=self.sr, 
                units='time',
                hop_length=512,
                trim=True
            )
            
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
            else:
                tempo = float(tempo)
                
            if len(beats) < 4:
                print("Insufficient beats detected")
                return tempo, beats, []

            print(f"Detected {len(beats)} beats at {tempo:.1f} BPM")
            
            # Detect time signature and downbeats
            time_signature, downbeat_indices = self._detect_time_signature_and_downbeats(beats)
            
            # Build measures using detected downbeats
            measures = self._build_flexible_measures(beats, downbeat_indices, time_signature)
            
            print(f"Detected {time_signature} time signature with {len(measures)} measures")
            return tempo, beats, measures

        except Exception as e:
            print(f"Beat detection failed: {e}")
            return 120.0, [], []

    def _detect_time_signature_and_downbeats(self, beats):
        """
        Detect time signature (3/4 or 4/4) and find downbeat positions
        """
        # Use onset strength to find strong beats (likely downbeats)
        onset_strength = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=self.sr)
        
        # Get strength at each beat position
        beat_strengths = []
        for beat_time in beats:
            closest_idx = np.argmin(np.abs(onset_times - beat_time))
            beat_strengths.append(onset_strength[closest_idx])
        
        # Try both 3/4 and 4/4 time signatures to see which fits better
        scores_3_4 = self._score_time_signature(beat_strengths, 3)
        scores_4_4 = self._score_time_signature(beat_strengths, 4)
        
        if scores_4_4['confidence'] > scores_3_4['confidence']:
            time_signature = "4/4"
            beats_per_measure = 4
            best_offset = scores_4_4['offset']
        else:
            time_signature = "3/4"
            beats_per_measure = 3
            best_offset = scores_3_4['offset']
        
        # Generate downbeat indices starting from the best offset
        downbeat_indices = list(range(best_offset, len(beats), beats_per_measure))
        
        print(f"Time signature: {time_signature}, first downbeat at beat {best_offset + 1}")
        return time_signature, downbeat_indices

    def _score_time_signature(self, beat_strengths, beats_per_measure):
        """
        Score how well a given time signature fits the beat strength pattern
        """
        best_score = 0
        best_offset = 0
        
        for offset in range(min(beats_per_measure, len(beat_strengths))):
            # Get strengths of beats that would be downbeats with this offset
            downbeat_strengths = []
            other_beat_strengths = []
            
            for i, strength in enumerate(beat_strengths):
                if (i - offset) % beats_per_measure == 0 and i >= offset:
                    downbeat_strengths.append(strength)
                elif i >= offset:
                    other_beat_strengths.append(strength)
            
            if len(downbeat_strengths) >= 2 and len(other_beat_strengths) >= 2:
                # Calculate how much stronger downbeats are compared to other beats
                avg_downbeat = np.mean(downbeat_strengths)
                avg_other = np.mean(other_beat_strengths)
                strength_ratio = avg_downbeat / (avg_other + 1e-10)
                
                # Also consider consistency of downbeat strengths
                downbeat_consistency = 1.0 / (1.0 + np.std(downbeat_strengths))
                
                # Combined score
                score = strength_ratio * downbeat_consistency
                
                if score > best_score:
                    best_score = score
                    best_offset = offset
        
        return {
            'confidence': best_score,
            'offset': best_offset
        }

    def _build_flexible_measures(self, beats, downbeat_indices, time_signature):
        """
        Build measures that can handle irregular timing and pauses
        """
        measures = []
        beats_per_measure = 4 if time_signature == "4/4" else 3
        
        # Calculate average beat duration for proper measure endings
        beat_intervals = np.diff(beats)
        avg_beat_duration = np.median(beat_intervals) if len(beat_intervals) > 0 else 0.5
        
        for i, downbeat_idx in enumerate(downbeat_indices):
            # FIXED: Always limit to the expected number of beats per measure
            remaining_beats = len(beats) - downbeat_idx
            actual_beats_in_measure = min(beats_per_measure, remaining_beats)
            
            # Create measure with exactly the right number of beats
            measure_beat_indices = list(range(downbeat_idx, downbeat_idx + actual_beats_in_measure))
            
            if len(measure_beat_indices) == 0:
                continue
                
            # Get the actual beat times for this measure
            measure_beats = [beats[idx] for idx in measure_beat_indices if idx < len(beats)]
            
            if len(measure_beats) >= 2:  # At least 2 beats for a valid measure
                # Calculate proper measure boundaries
                measure_start = measure_beats[0]  # Start at first beat (downbeat)
                
                # End time: if there's a next measure, end at its downbeat
                # Otherwise, end after the duration of the last beat
                if i + 1 < len(downbeat_indices) and downbeat_indices[i + 1] < len(beats):
                    measure_end = beats[downbeat_indices[i + 1]]
                else:
                    measure_end = measure_beats[-1] + avg_beat_duration
                
                # Check for irregular timing (long pauses)
                intervals = np.diff(measure_beats)
                median_interval = np.median(intervals) if len(intervals) > 0 else 0.5
                has_pause = any(interval > 3 * median_interval for interval in intervals)
                
                measures.append({
                    'start': measure_start,
                    'end': measure_end,
                    'beats': measure_beats,
                    'time_signature': time_signature,
                    'beats_per_measure': len(measure_beats),  # This should now be 3 or 4
                    'has_irregular_timing': has_pause or len(measure_beats) != beats_per_measure,
                    'measure_number': len(measures) + 1
                })
        
        return measures

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
        Detect guitar licks by evaluating each measure individually
        If a measure has significant guitar activity, mark it as a lick
        This ensures licks always start on downbeats and eliminates gaps
        """
        # Get beats and measures
        tempo, beats, measures = self.detect_beats_and_measures()
        
        if len(measures) < min_lick_measures:
            print(f"Not enough measures ({len(measures)}) for lick detection")
            return []
        
        print(f"Analyzing {len(measures)} measures individually for guitar activity")
        
        # Get musical activity across the audio
        activity_timeline = self._detect_musical_activity()
        
        # Get onsets for counting note activity in licks
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr, units='time', hop_length=512)
        
        # Evaluate each measure individually
        measure_activities = []
        for i, measure in enumerate(measures):
            start_time = measure['start']
            end_time = measure['end']
            
            # Calculate activity score for this measure
            activity_score = self._calculate_region_activity(activity_timeline, start_time, end_time)
            
            # Count onsets in this measure
            measure_onsets = [o for o in onsets if start_time <= o <= end_time]
            onset_count = len(measure_onsets)
            
            measure_activities.append({
                'measure_index': i,
                'measure': measure,
                'activity_score': activity_score,
                'onset_count': onset_count,
                'is_active': activity_score > 0.3  # Threshold for guitar activity
            })
        
        # Now create licks from consecutive active measures
        licks = []
        i = 0
        
        while i < len(measure_activities):
            if measure_activities[i]['is_active']:
                # Start a new lick from this measure
                lick_measures = [measure_activities[i]]
                j = i + 1
                
                # Try to extend the lick with consecutive active measures (up to max_lick_measures)
                while (j < len(measure_activities) and 
                       measure_activities[j]['is_active'] and 
                       len(lick_measures) < max_lick_measures):
                    lick_measures.append(measure_activities[j])
                    j += 1
                
                # Only create lick if it meets minimum length requirement
                if len(lick_measures) >= min_lick_measures:
                    # Calculate lick boundaries - always measure boundaries
                    start_time = lick_measures[0]['measure']['start']
                    end_time = lick_measures[-1]['measure']['end']
                    
                    # Collect all beat positions for this lick
                    beat_positions = []
                    total_beats = 0
                    total_onsets = 0
                    measure_numbers = []
                    
                    for lick_measure in lick_measures:
                        measure = lick_measure['measure']
                        beat_positions.extend(measure['beats'])
                        total_beats += measure['beats_per_measure']
                        total_onsets += lick_measure['onset_count']
                        measure_numbers.append(measure['measure_number'])
                    
                    # Calculate average activity score
                    avg_activity = sum(lm['activity_score'] for lm in lick_measures) / len(lick_measures)
                    
                    licks.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time,
                        'measures': len(lick_measures),
                        'beats': total_beats,
                        'onset_count': total_onsets,
                        'beat_positions': beat_positions,
                        'activity_score': avg_activity,
                        'measure_numbers': measure_numbers,
                        'time_signature': lick_measures[0]['measure']['time_signature'],
                        'has_irregular_timing': any(lm['measure']['has_irregular_timing'] for lm in lick_measures)
                    })
                
                # Move to the end of this lick
                i = j
            else:
                # This measure is not active, move to next
                i += 1
        
        print(f"Detected {len(licks)} guitar licks from measure-by-measure analysis")
        return licks

    def _detect_musical_activity(self):
        """
        Detect overall musical activity using onset density and spectral features
        Much simpler than the previous guitar-specific detection
        """
        # Detect onsets
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr, units='time', hop_length=512)
        
        # Get spectral features that indicate musical activity
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        rms_energy = librosa.feature.rms(y=self.y)[0]
        
        # Create timeline with 0.5 second windows
        window_size = 0.5
        duration = len(self.y) / self.sr
        timeline = []
        
        for t in np.arange(0, duration - window_size, window_size / 2):  # 50% overlap
            window_start = t
            window_end = t + window_size
            
            # Count onsets in this window
            window_onsets = len([o for o in onsets if window_start <= o <= window_end])
            onset_density = window_onsets / window_size
            
            # Get average spectral features for this window
            start_frame = int(window_start * self.sr / 512)
            end_frame = int(window_end * self.sr / 512)
            
            if start_frame < len(spectral_centroid) and end_frame <= len(spectral_centroid):
                avg_centroid = np.mean(spectral_centroid[start_frame:end_frame])
                avg_rolloff = np.mean(spectral_rolloff[start_frame:end_frame])
                avg_energy = np.mean(rms_energy[start_frame:end_frame])
                
                # Simple activity score based on onset density, spectral brightness, and energy
                activity_score = (
                    min(onset_density / 10.0, 0.4) +  # Onset contribution (max 0.4)
                    min(avg_centroid / 10000.0, 0.3) +  # Brightness contribution (max 0.3)
                    min(avg_energy * 10, 0.3)  # Energy contribution (max 0.3)
                )
                
                timeline.append({
                    'time': t + window_size / 2,
                    'activity': min(activity_score, 1.0)
                })
        
        return timeline

    def _calculate_region_activity(self, activity_timeline, start_time, end_time):
        """
        Calculate average activity score for a time region
        """
        region_activities = [
            entry['activity'] for entry in activity_timeline 
            if start_time <= entry['time'] <= end_time
        ]
        
        return np.mean(region_activities) if region_activities else 0.0

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
        
        def to_native(val):
            # Convert numpy types to native Python types
            if isinstance(val, (np.generic,)):
                return val.item()
            if isinstance(val, (list, tuple)):
                return [to_native(v) for v in val]
            if isinstance(val, dict):
                return {k: to_native(v) for k, v in val.items()}
            return val
        
        # Prepare data for web interface
        dashboard_data = {
            'audio_file': str(os.path.basename(audio_filename)),
            'tempo': float(results.get('tempo', 120)),
            'total_duration': float(self.duration),
            'licks': []
        }
        
        for lick in results['analysis']:
            lick_data = {
                'id': int(lick['lick_number']),
                'start_time': float(round(lick['start_time'], 2)),
                'end_time': float(round(lick['end_time'], 2)),
                'duration': float(round(lick['duration'], 2)),
                'beats': int(lick['beats']),
                'onset_count': int(lick['onset_count']),
                'activity_score': float(round(lick['activity_score'], 2)),
                'primary_chord': str(lick['primary_chord']),
                'all_chords': [str(c['chord']) for c in lick['chords']],
                'beat_positions': [float(round(pos, 2)) for pos in lick['beat_positions']]
            }
            dashboard_data['licks'].append(to_native(lick_data))
        
        # Save JSON file
        with open(output_json, 'w') as f:
            json.dump(to_native(dashboard_data), f, indent=2)
        
        print(f"\nDashboard data exported to {output_json}")
        return dashboard_data

    def detect_drum_hits(self):
        """
        Detect individual drum hits (kick, snare, hi-hat) using spectral analysis
        """
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(self.y, margin=8)
        
        # Detect onsets in percussive component only
        onset_frames = librosa.onset.onset_detect(
            y=percussive, 
            sr=self.sr, 
            units='time',
            hop_length=256,  # Higher resolution
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.1,
            wait=0.05  # Minimum time between onsets
        )
        
        if len(onset_frames) == 0:
            return {'kicks': [], 'snares': [], 'hihats': []}
        
        # Analyze each onset to classify drum type
        kicks = []
        snares = []
        hihats = []
        
        for onset_time in onset_frames:
            # Extract a small window around the onset
            start_sample = max(0, int((onset_time - 0.01) * self.sr))
            end_sample = min(len(percussive), int((onset_time + 0.1) * self.sr))
            onset_audio = percussive[start_sample:end_sample]
            
            if len(onset_audio) < 256:
                continue
            
            # Compute spectral features for classification
            spectral_centroid = librosa.feature.spectral_centroid(y=onset_audio, sr=self.sr)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(onset_audio)[0]
            avg_zcr = np.mean(zero_crossing_rate)
            
            # Get frequency spectrum
            fft = np.abs(np.fft.fft(onset_audio))
            freqs = np.fft.fftfreq(len(onset_audio), 1/self.sr)
            
            # Energy in different frequency bands
            low_energy = np.sum(fft[(freqs >= 40) & (freqs <= 150)]**2)  # Kick range
            mid_energy = np.sum(fft[(freqs >= 150) & (freqs <= 400)]**2)  # Snare fundamentals
            high_energy = np.sum(fft[(freqs >= 1000) & (freqs <= 8000)]**2)  # Snare harmonics/hi-hat
            very_high_energy = np.sum(fft[(freqs >= 8000) & (freqs <= 15000)]**2)  # Hi-hat
            
            total_energy = low_energy + mid_energy + high_energy + very_high_energy
            
            if total_energy == 0:
                continue
            
            # Normalize energy ratios
            low_ratio = low_energy / total_energy
            high_ratio = high_energy / total_energy
            very_high_ratio = very_high_energy / total_energy
            
            # Classify based on spectral characteristics
            if low_ratio > 0.4 and avg_centroid < 200:
                # High low-frequency energy, low centroid = kick drum
                kicks.append({
                    'time': onset_time,
                    'confidence': low_ratio,
                    'centroid': avg_centroid
                })
            elif high_ratio > 0.3 and 200 < avg_centroid < 3000 and avg_zcr > 0.1:
                # Balanced mid/high energy with moderate centroid = snare
                snares.append({
                    'time': onset_time,
                    'confidence': high_ratio,
                    'centroid': avg_centroid,
                    'zcr': avg_zcr
                })
            elif very_high_ratio > 0.2 and avg_centroid > 3000:
                # Very high frequency energy = hi-hat
                hihats.append({
                    'time': onset_time,
                    'confidence': very_high_ratio,
                    'centroid': avg_centroid
                })
        
        print(f"Detected drums: {len(kicks)} kicks, {len(snares)} snares, {len(hihats)} hi-hats")
        return {
            'kicks': kicks,
            'snares': snares,
            'hihats': hihats
        }

    def detect_beats_and_measures_drum_aware(self):
        """
        Detect beats and measures using drum pattern analysis
        Handles variable measure lengths and irregular timing
        """
        try:
            # Get basic tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, units='time')
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
            else:
                tempo = float(tempo)
            
            if len(beats) < 2:
                return tempo, beats, []
            
            # Detect drum hits
            drum_hits = self.detect_drum_hits()
            
            # If we have snare hits, use them to establish the beat grid
            if len(drum_hits['snares']) >= 2:
                print("Using snare-based beat alignment...")
                return self._align_beats_with_snares(tempo, beats, drum_hits)
            elif len(drum_hits['kicks']) >= 2:
                print("Using kick-based beat alignment...")
                return self._align_beats_with_kicks(tempo, beats, drum_hits)
            else:
                print("Falling back to onset-based beat alignment...")
                return self._align_beats_with_onsets(tempo, beats)
                
        except Exception as e:
            print(f"Drum-aware beat detection failed: {e}")
            return 120.0, [], []

    def _align_beats_with_snares(self, tempo, beats, drum_hits):
        """
        Use snare hits (typically on beats 2 and 4) to establish proper beat alignment
        """
        snare_times = [hit['time'] for hit in drum_hits['snares']]
        
        # Find which detected beats are closest to snare hits
        snare_beat_indices = []
        for snare_time in snare_times:
            closest_beat_idx = np.argmin(np.abs(beats - snare_time))
            if abs(beats[closest_beat_idx] - snare_time) < 0.2:  # Within 200ms
                snare_beat_indices.append(closest_beat_idx)
        
        if len(snare_beat_indices) < 2:
            return self._align_beats_with_onsets(tempo, beats)
        
        # Analyze the pattern of snare beats to determine time signature
        snare_intervals = []
        for i in range(1, len(snare_beat_indices)):
            interval = snare_beat_indices[i] - snare_beat_indices[i-1]
            snare_intervals.append(interval)
        
        # Most common interval between snares
        most_common_interval = max(set(snare_intervals), key=snare_intervals.count)
        
        # In 4/4 time, snares are typically 2 beats apart (beat 2 to beat 4, or beat 4 to beat 2 of next measure)
        if most_common_interval == 2:
            beats_per_measure = 4
            print("Detected 4/4 time signature")
        elif most_common_interval == 3:
            beats_per_measure = 6  # Could be 6/8 or 6/4
            print("Detected 6/8 or 6/4 time signature")
        elif most_common_interval == 4:
            beats_per_measure = 8  # Could be 8/8 time
            print("Detected 8/8 time signature")
        else:
            beats_per_measure = 4  # Default to 4/4
            print(f"Unusual snare pattern (interval={most_common_interval}), defaulting to 4/4")
        
        # Find the first downbeat by working backwards from the first snare
        first_snare_idx = snare_beat_indices[0]
        
        # In 4/4, if snare is on beat 2, downbeat is 1 beat earlier
        # In 4/4, if snare is on beat 4, downbeat is 3 beats earlier
        # Try both possibilities and see which gives more consistent patterns
        possible_downbeat_offsets = []
        if beats_per_measure == 4:
            possible_downbeat_offsets = [-1, -3]  # Beat 2 or beat 4
        elif beats_per_measure == 6:
            possible_downbeat_offsets = [-1, -3, -5]  # Various possibilities in 6/8
        else:
            possible_downbeat_offsets = [-1, -3]  # Default
        
        best_downbeat_idx = first_snare_idx - 1  # Default assumption
        best_score = 0
        
        for offset in possible_downbeat_offsets:
            potential_downbeat_idx = first_snare_idx + offset
            if potential_downbeat_idx < 0:
                continue
            
            # Score this downbeat position by checking how well subsequent snares align
            score = 0
            for snare_idx in snare_beat_indices:
                beats_from_downbeat = snare_idx - potential_downbeat_idx
                if beats_from_downbeat > 0:
                    # Check if this snare falls on expected beats (2, 4, 6, etc.)
                    expected_snare_positions = list(range(1, beats_per_measure, 2))  # 1, 3, 5... (0-indexed as beat 2, 4, 6...)
                    if beats_per_measure == 4:
                        expected_snare_positions = [1, 3]  # Beat 2 and 4
                    elif beats_per_measure == 6:
                        expected_snare_positions = [1, 3, 5]  # Beat 2, 4, 6
                    
                    beat_in_measure = beats_from_downbeat % beats_per_measure
                    if beat_in_measure in expected_snare_positions:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_downbeat_idx = potential_downbeat_idx
        
        # Generate measures starting from the best downbeat
        measures = []
        i = best_downbeat_idx
        
        while i + beats_per_measure - 1 < len(beats):
            measure_beats = beats[i:i + beats_per_measure]
            measures.append({
                'start': measure_beats[0],
                'end': measure_beats[-1],
                'beats': measure_beats.tolist(),
                'time_signature': f"{beats_per_measure}/4"
            })
            i += beats_per_measure
        
        print(f"Snare-aligned {len(measures)} measures ({beats_per_measure}/4 time)")
        return tempo, beats, measures

    def _align_beats_with_kicks(self, tempo, beats, drum_hits):
        """
        Use kick hits (typically on beats 1 and 3) to establish beat alignment
        """
        kick_times = [hit['time'] for hit in drum_hits['kicks']]
        
        # Find which detected beats are closest to kick hits
        kick_beat_indices = []
        for kick_time in kick_times:
            closest_beat_idx = np.argmin(np.abs(beats - kick_time))
            if abs(beats[closest_beat_idx] - kick_time) < 0.2:  # Within 200ms
                kick_beat_indices.append(closest_beat_idx)
        
        if len(kick_beat_indices) < 2:
            return self._align_beats_with_onsets(tempo, beats)
        
        # In 4/4, kicks are typically on beats 1 and 3 (2 beats apart)
        # Use the first kick as a likely downbeat
        first_kick_idx = kick_beat_indices[0]
        beats_per_measure = 4  # Assume 4/4 for kick-based detection
        
        measures = []
        i = first_kick_idx
        
        while i + beats_per_measure - 1 < len(beats):
            measure_beats = beats[i:i + beats_per_measure]
            measures.append({
                'start': measure_beats[0],
                'end': measure_beats[-1],
                'beats': measure_beats.tolist(),
                'time_signature': "4/4"
            })
            i += beats_per_measure
        
        print(f"Kick-aligned {len(measures)} measures (4/4 time)")
        return tempo, beats, measures

    def _align_beats_with_onsets(self, tempo, beats):
        """
        Fallback method using onset strength analysis
        """
        # Use the original onset strength method but with dynamic measure detection
        onset_strength = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=self.sr)
        
        # Find the strongest beats (likely downbeats)
        beat_strengths = []
        for beat_time in beats:
            closest_idx = np.argmin(np.abs(onset_times - beat_time))
            beat_strengths.append(onset_strength[closest_idx])
        
        # Find peaks in beat strength that might indicate downbeats
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(beat_strengths, height=np.percentile(beat_strengths, 75))
        
        if len(peaks) >= 2:
            # Use intervals between strong beats to estimate measure length
            peak_intervals = np.diff(peaks)
            most_common_interval = max(set(peak_intervals), key=list(peak_intervals).count) if len(peak_intervals) > 0 else 4
            beats_per_measure = int(most_common_interval)
        else:
            beats_per_measure = 4  # Default
        
        # Start from first strong beat
        first_downbeat_idx = peaks[0] if len(peaks) > 0 else 0
        
        measures = []
        i = first_downbeat_idx
        
        while i + beats_per_measure - 1 < len(beats):
            measure_beats = beats[i:i + beats_per_measure]
            measures.append({
                'start': measure_beats[0],
                'end': measure_beats[-1],
                'beats': measure_beats.tolist(),
                'time_signature': f"{beats_per_measure}/4"
            })
            i += beats_per_measure
        
        print(f"Onset-aligned {len(measures)} measures ({beats_per_measure}/4 time)")
        return tempo, beats, measures

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GuitarSoloAnalyzer("Bertha (Veneta 72).mp3")
    
    # Analyze guitar licks - focus on the solo section with measure-aligned licks
    results = analyzer.analyze_guitar_licks(
        start_time=120, 
        end_time=300,
        min_lick_measures=1,  # Minimum 1 measure (4 beats)
        max_lick_measures=2,   # Maximum 2 measures (8 beats)
    )
    
    # Print results
    analyzer.print_lick_analysis(results)
    
    # Save to CSV with the new method
    analyzer.save_lick_analysis_to_csv(results, "guitar_licks.csv")
    
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