"""MIDI loading, melody extraction, and fingerprinting utilities."""

import pretty_midi


def load_midi(filepath):
    """Load a MIDI file and return the PrettyMIDI object."""
    return pretty_midi.PrettyMIDI(filepath)


def extract_melody_skyline(midi, time_step=0.05):
    """Extract a monophonic melody using the skyline algorithm.
    
    At each small time window, pick the highest-pitched note that's sounding.
    Returns a list of (time, pitch) tuples.
    """
    all_notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)
    
    if not all_notes:
        return []
    
    # Sort by start time
    all_notes.sort(key=lambda n: n.start)
    end_time = max(n.end for n in all_notes)
    
    melody = []
    last_pitch = None
    
    # Active notes: list of notes currently sounding
    active = []
    note_idx = 0
    
    t = 0.0
    while t < end_time:
        # Add notes that have started by time t
        while note_idx < len(all_notes) and all_notes[note_idx].start <= t:
            active.append(all_notes[note_idx])
            note_idx += 1
        
        # Remove notes that have ended by time t
        active = [n for n in active if n.end > t]
        
        if active:
            highest = max(active, key=lambda n: n.pitch)
            if highest.pitch != last_pitch:
                melody.append((t, highest.pitch))
                last_pitch = highest.pitch
        else:
            last_pitch = None
        
        t += time_step
    
    return melody


def melody_to_intervals(melody):
    """Convert melody [(t, pitch), ...] to a list of pitch intervals (semitones)."""
    if len(melody) < 2:
        return []
    return [melody[i][1] - melody[i-1][1] for i in range(1, len(melody))]


def midi_to_fingerprint(filepath):
    """Full pipeline: MIDI file path → interval sequence (fingerprint).
    
    Returns None if the file can't be loaded or has no usable notes.
    """
    try:
        midi = load_midi(filepath)
    except Exception:
        return None
    
    melody = extract_melody_skyline(midi)
    if len(melody) < 2:
        return None
    
    return melody_to_intervals(melody)
