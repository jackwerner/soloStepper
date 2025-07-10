import streamlit as st
import re
import streamlit.components.v1 as components

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def create_youtube_player(video_id, start_time, end_time):
    """Create YouTube player with looping functionality"""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #player {{
                width: 100%;
                height: 400px;
            }}
            .controls {{
                margin: 10px 0;
                text-align: center;
            }}
            .time-display {{
                font-family: monospace;
                font-size: 18px;
                margin: 10px 0;
                background: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
            }}
            button {{
                margin: 0 5px;
                padding: 8px 16px;
                font-size: 14px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                background: #ff4444;
                color: white;
            }}
            button:hover {{
                background: #cc3333;
            }}
            .loop-active {{
                background: #44ff44 !important;
                color: black !important;
            }}
            .status {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            .updated {{
                background: #ffffaa !important;
                transition: background 0.5s ease;
            }}
        </style>
    </head>
    <body>
        <div id="player"></div>
        
        <div class="time-display">
            <div>Current Time: <span id="currentTime">0.00</span>s</div>
            <div>Loop: <span id="loopStart">{start_time:.2f}</span>s - <span id="loopEnd">{end_time:.2f}</span>s</div>
            <div class="status" id="status">Click on timeline to set start time</div>
        </div>
        
        <div class="controls">
            <button id="playBtn" onclick="playVideo()">Play</button>
            <button id="pauseBtn" onclick="pauseVideo()">Pause</button>
            <button id="loopBtn" onclick="toggleLoop()" class="loop-active">Loop ON</button>
            <button onclick="jumpToStart()">Jump to Start</button>
            <button onclick="resetLoop()">Reset to Loop Start</button>
            <button onclick="setCurrentAsEnd()">Set Current as End</button>
        </div>

        <script>
            var tag = document.createElement('script');
            tag.src = "https://www.youtube.com/iframe_api";
            var firstScriptTag = document.getElementsByTagName('script')[0];
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

            var player;
            var loopActive = true;
            var startTime = {start_time};
            var endTime = {end_time};
            var checkInterval;
            var lastTime = 0;
            var seekInProgress = false;
            var seekTimeout;
            var allowLooping = true;
            var wasPlaying = false;

            function onYouTubeIframeAPIReady() {{
                player = new YT.Player('player', {{
                    height: '400',
                    width: '100%',
                    videoId: '{video_id}',
                    playerVars: {{
                        'playsinline': 1,
                        'rel': 0,
                        'showinfo': 0,
                        'modestbranding': 1
                    }},
                    events: {{
                        'onReady': onPlayerReady,
                        'onStateChange': onPlayerStateChange
                    }}
                }});
            }}

            function onPlayerReady(event) {{
                console.log('Player ready');
                jumpToStart();
                startTimeUpdater();
            }}

            function onPlayerStateChange(event) {{
                // Track if video was playing before seeking
                if (event.data == YT.PlayerState.PLAYING) {{
                    wasPlaying = true;
                    if (loopActive) {{
                        startLoopChecker();
                    }}
                }} else {{
                    if (event.data == YT.PlayerState.PAUSED) {{
                        wasPlaying = false;
                    }}
                    stopLoopChecker();
                }}
                
                // Update status
                var status = document.getElementById('status');
                switch(event.data) {{
                    case YT.PlayerState.PLAYING:
                        if (!seekInProgress) {{
                            status.textContent = 'Playing - Click timeline to set start time';
                        }}
                        break;
                    case YT.PlayerState.PAUSED:
                        status.textContent = 'Paused - Click timeline to set start time';
                        break;
                    case YT.PlayerState.BUFFERING:
                        status.textContent = 'Buffering...';
                        break;
                    default:
                        status.textContent = 'Click timeline to set start time';
                }}
            }}

            function detectUserSeek(currentTime) {{
                // Detect if user manually seeked (large time jump that's not our loop)
                var timeDiff = Math.abs(currentTime - lastTime);
                var isLargeJump = timeDiff > 1.0; // More than 1 second jump
                var isNotOurLoop = !(Math.abs(currentTime - startTime) < 0.5); // Not jumping to our loop start
                
                if (isLargeJump && isNotOurLoop && !seekInProgress) {{
                    // User manually seeked - set this as new start time!
                    seekInProgress = true;
                    allowLooping = false;
                    
                    // Update start time to current position
                    startTime = currentTime;
                    
                    // Update display
                    var loopStartSpan = document.getElementById('loopStart');
                    loopStartSpan.textContent = startTime.toFixed(2);
                    
                    // Visual feedback
                    var timeDisplay = document.querySelector('.time-display');
                    timeDisplay.classList.add('updated');
                    setTimeout(function() {{
                        timeDisplay.classList.remove('updated');
                    }}, 1000);
                    
                    document.getElementById('status').textContent = `Start time set to ${{startTime.toFixed(2)}}s`;
                    
                    // Clear any existing timeout
                    if (seekTimeout) {{
                        clearTimeout(seekTimeout);
                    }}
                    
                    // Re-enable looping after 2 seconds
                    seekTimeout = setTimeout(function() {{
                        seekInProgress = false;
                        allowLooping = true;
                        document.getElementById('status').textContent = 'Loop ready - Click timeline to update start time';
                    }}, 2000);
                }}
                
                lastTime = currentTime;
            }}

            function startTimeUpdater() {{
                setInterval(function() {{
                    if (player && player.getCurrentTime) {{
                        var currentTime = player.getCurrentTime();
                        document.getElementById('currentTime').textContent = currentTime.toFixed(2);
                        
                        // Check for manual seeks
                        detectUserSeek(currentTime);
                    }}
                }}, 100);
            }}

            function startLoopChecker() {{
                stopLoopChecker();
                checkInterval = setInterval(function() {{
                    if (player && player.getCurrentTime && allowLooping) {{
                        var currentTime = player.getCurrentTime();
                        if (currentTime >= endTime && loopActive && !seekInProgress) {{
                            player.seekTo(startTime, true);
                            lastTime = startTime; // Update lastTime to prevent false seek detection
                        }}
                    }}
                }}, 150);
            }}

            function stopLoopChecker() {{
                if (checkInterval) {{
                    clearInterval(checkInterval);
                    checkInterval = null;
                }}
            }}

            function playVideo() {{
                if (player && player.playVideo) {{
                    player.playVideo();
                }}
            }}

            function pauseVideo() {{
                if (player && player.pauseVideo) {{
                    player.pauseVideo();
                }}
            }}

            function toggleLoop() {{
                loopActive = !loopActive;
                var btn = document.getElementById('loopBtn');
                if (loopActive) {{
                    btn.textContent = 'Loop ON';
                    btn.classList.add('loop-active');
                    if (player && player.getPlayerState() == YT.PlayerState.PLAYING) {{
                        startLoopChecker();
                    }}
                }} else {{
                    btn.textContent = 'Loop OFF';
                    btn.classList.remove('loop-active');
                    stopLoopChecker();
                }}
            }}

            function jumpToStart() {{
                if (player && player.seekTo) {{
                    seekInProgress = true;
                    allowLooping = false;
                    player.seekTo(startTime, true);
                    lastTime = startTime;
                    
                    setTimeout(function() {{
                        seekInProgress = false;
                        allowLooping = true;
                    }}, 1000);
                }}
            }}

            function resetLoop() {{
                if (player) {{
                    seekInProgress = true;
                    allowLooping = false;
                    player.seekTo(startTime, true);
                    lastTime = startTime;
                    
                    if (player.getPlayerState() != YT.PlayerState.PLAYING) {{
                        player.playVideo();
                    }}
                    
                    setTimeout(function() {{
                        seekInProgress = false;
                        allowLooping = true;
                    }}, 1000);
                }}
            }}

            function setCurrentAsEnd() {{
                if (player && player.getCurrentTime) {{
                    var currentTime = player.getCurrentTime();
                    if (currentTime > startTime) {{
                        endTime = currentTime;
                        document.getElementById('loopEnd').textContent = endTime.toFixed(2);
                        
                        // Visual feedback
                        var timeDisplay = document.querySelector('.time-display');
                        timeDisplay.classList.add('updated');
                        setTimeout(function() {{
                            timeDisplay.classList.remove('updated');
                        }}, 1000);
                        
                        document.getElementById('status').textContent = `End time set to ${{endTime.toFixed(2)}}s`;
                        
                        setTimeout(function() {{
                            document.getElementById('status').textContent = 'Loop updated! Click timeline to change start time';
                        }}, 2000);
                    }} else {{
                        document.getElementById('status').textContent = 'End time must be after start time!';
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html_code

def main():
    st.set_page_config(
        page_title="YouTube Loop Player", 
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 YouTube Loop Player")
    st.markdown("Perfect for practicing music sections with precise timing!")
    
    # Initialize session state for tracking start time changes
    if 'last_start_time' not in st.session_state:
        st.session_state.last_start_time = 0.0
    if 'default_duration' not in st.session_state:
        st.session_state.default_duration = 10.0
    if 'auto_end_time' not in st.session_state:
        st.session_state.auto_end_time = 10.0
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        youtube_url = st.text_input(
            "YouTube URL:", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube URL here"
        )
    
    with col2:
        st.markdown("### Examples:")
        st.markdown("- youtube.com/watch?v=abc123")
        st.markdown("- youtu.be/abc123")
        st.markdown("- youtube.com/embed/abc123")
    
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if video_id:
            st.success(f"✅ Video ID extracted: {video_id}")
            
            # Time controls
            st.markdown("### ⏰ Loop Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_time_total = st.number_input(
                    "Start (seconds):", 
                    min_value=0.0, 
                    value=0.0, 
                    step=0.01, 
                    format="%.2f",
                    key="start_time"
                )
                
                # Auto-adjust default duration based on user's typical usage
                duration_preset = st.selectbox(
                    "Default duration:",
                    [5.0, 8.0, 10.0, 15.0, 20.0, 30.0],
                    index=2,  # Default to 10 seconds
                    format_func=lambda x: f"{x:.0f}s"
                )
                st.session_state.default_duration = duration_preset
                
                # Check if start time changed and update auto end time
                if start_time_total != st.session_state.last_start_time:
                    st.session_state.auto_end_time = start_time_total + st.session_state.default_duration
                    st.session_state.last_start_time = start_time_total
            
            with col2:
                # Use the auto-calculated end time as default
                end_time_total = st.number_input(
                    "End (seconds):", 
                    min_value=0.0, 
                    value=st.session_state.auto_end_time, 
                    step=0.01, 
                    format="%.2f",
                    key="end_time"
                )
            
            with col3:
                st.markdown("**Loop Info:**")
                st.write(f"Start: {start_time_total:.2f}s")
                st.write(f"End: {end_time_total:.2f}s")
                duration = end_time_total - start_time_total
                st.write(f"Duration: {duration:.2f}s")
                
                # Show time in MM:SS format for longer durations
                if start_time_total >= 60:
                    start_mins = int(start_time_total // 60)
                    start_secs = start_time_total % 60
                    st.write(f"Start: {start_mins}:{start_secs:05.2f}")
                
                if end_time_total >= 60:
                    end_mins = int(end_time_total // 60)
                    end_secs = end_time_total % 60
                    st.write(f"End: {end_mins}:{end_secs:05.2f}")
            
            if end_time_total > start_time_total:
                st.markdown("### 🎬 Video Player")
                
                # Create and display the player
                player_html = create_youtube_player(video_id, start_time_total, end_time_total)
                components.html(player_html, height=600)
                
                st.markdown("### 🎮 How to Use:")
                st.markdown(f"""
                - **🎯 Click timeline** to set start time (end auto-sets to +{st.session_state.default_duration:.0f}s)
                - **▶️ Play/Pause**: Basic video controls  
                - **🔄 Loop ON/OFF**: Toggle automatic looping
                - **⏭️ Jump to Start**: Go to loop start time
                - **🔄 Reset**: Jump to start and play
                - **⏹️ Set Current as End**: Use current position as end time
                
                **Pro tip**: Change "Default duration" above to customize auto-end timing!
                """)
                
                # Quick time presets
                st.markdown("### ⚡ Quick Presets")
                col1, col2, col3, col4 = st.columns(4)
                
                preset_data = [
                    ("🎵 8s from start", 0, 8),
                    ("🎼 16s from start", 0, 16), 
                    ("🎸 30s from start", 0, 30),
                    ("🥁 1min from start", 0, 60)
                ]
                
                for i, (label, start, end) in enumerate(preset_data):
                    with [col1, col2, col3, col4][i]:
                        if st.button(label, key=f"preset_{i}"):
                            # Use session state keys that don't conflict with widget keys
                            st.session_state.last_start_time = start - 1  # Force update
                            st.session_state.auto_end_time = end
                            st.rerun()
                
            else:
                st.error("❌ End time must be greater than start time!")
                
        else:
            st.error("❌ Invalid YouTube URL. Please check the format.")
    
    else:
        st.info("👆 Enter a YouTube URL above to get started!")
        
        # Show example
        st.markdown("### 🎯 Perfect for:")
        st.markdown("""
        - **Music practice**: Loop guitar solos, drum fills, vocal runs
        - **Dance practice**: Repeat choreography sections  
        - **Language learning**: Replay pronunciation examples
        - **Tutorials**: Focus on specific technique demonstrations
        """)

if __name__ == "__main__":
    main()
