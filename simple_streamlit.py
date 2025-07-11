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
            .set-button {{
                background: #4444ff !important;
            }}
            .set-button:hover {{
                background: #3333cc !important;
            }}
            .save-section {{
                margin: 20px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }}
            .save-controls {{
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            .save-controls input {{
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }}
            .save-button {{
                background: #28a745 !important;
                color: white;
            }}
            .save-button:hover {{
                background: #218838 !important;
            }}
            .saved-loops {{
                margin-top: 15px;
            }}
            .loop-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px;
                margin: 5px 0;
                background: white;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .loop-info {{
                flex: 1;
                text-align: left;
            }}
            .loop-actions {{
                display: flex;
                gap: 5px;
            }}
            .load-button {{
                background: #007bff !important;
                font-size: 12px;
                padding: 4px 8px;
            }}
            .load-button:hover {{
                background: #0056b3 !important;
            }}
            .delete-button {{
                background: #dc3545 !important;
                font-size: 12px;
                padding: 4px 8px;
            }}
            .delete-button:hover {{
                background: #c82333 !important;
            }}
        </style>
    </head>
    <body>
        <div id="player"></div>
        
        <div class="time-display">
            <div>Current Time: <span id="currentTime">0.00</span>s</div>
            <div>Loop: <span id="loopStart">{start_time:.2f}</span>s - <span id="loopEnd">{end_time:.2f}</span>s</div>
            <div class="status" id="status">Click timeline to set start time</div>
        </div>
        
        <div class="controls">
            <button id="playBtn" onclick="playVideo()">Play</button>
            <button id="pauseBtn" onclick="pauseVideo()">Pause</button>
            <button id="loopBtn" onclick="toggleLoop()" class="loop-active">Loop ON</button>
            <button onclick="jumpToStart()">Jump to Start</button>
            <button onclick="resetLoop()">Reset to Loop Start</button>
        </div>
        
        <div class="controls">
            <button class="set-button" onclick="setCurrentAsStart()">📍 Set Current as Start</button>
            <button class="set-button" onclick="setCurrentAsEnd()">🏁 Set Current as End</button>
        </div>

        <div class="save-section">
            <h3>💾 Save & Load Loops</h3>
            <div class="save-controls">
                <input type="text" id="loopName" placeholder="Loop name..." value="Loop 1">
                <button class="save-button" onclick="saveCurrentLoop()">💾 Save Current Loop</button>
                <span id="saveStatus" style="color: green; font-weight: bold;"></span>
            </div>
            <div id="savedLoops" class="saved-loops"></div>
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
            var savedLoops = [];
            var loopCounter = 1;

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
                displaySavedLoops();
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
                            status.textContent = 'Playing - Use buttons or click timeline to set loop points';
                        }}
                        break;
                    case YT.PlayerState.PAUSED:
                        status.textContent = 'Paused - Use buttons or click timeline to set loop points';
                        break;
                    case YT.PlayerState.BUFFERING:
                        status.textContent = 'Buffering...';
                        break;
                    default:
                        status.textContent = 'Use buttons or click timeline to set loop points';
                }}
            }}

            function detectUserSeek(currentTime) {{
                // Detect if user manually seeked (large time jump that's not our loop)
                var timeDiff = Math.abs(currentTime - lastTime);
                var isLargeJump = timeDiff > 1.0; // More than 1 second jump
                var isNotOurLoop = !(Math.abs(currentTime - startTime) < 0.5); // Not jumping to our loop start
                
                // ADDITIONAL CHECK: Don't auto-set if we're close to end time (likely a loop artifact)
                var isNearEndTime = Math.abs(currentTime - endTime) < 0.5;
                
                // ADDITIONAL CHECK: Don't auto-set if we just came from near the end (loop scenario)
                var wasNearEnd = Math.abs(lastTime - endTime) < 1.0;
                
                if (isLargeJump && isNotOurLoop && !seekInProgress && !isNearEndTime && !wasNearEnd) {{
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
                        document.getElementById('status').textContent = 'Loop ready - Use buttons or timeline to adjust';
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
                            seekInProgress = true; // Prevent detectUserSeek from interfering
                            player.seekTo(startTime, true);
                            lastTime = startTime; // Update lastTime to prevent false seek detection
                            
                            // Clear seek flag after a brief delay
                            setTimeout(function() {{
                                seekInProgress = false;
                            }}, 300);
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

            function setCurrentAsStart() {{
                if (player && player.getCurrentTime) {{
                    var currentTime = player.getCurrentTime();
                    startTime = currentTime;
                    document.getElementById('loopStart').textContent = startTime.toFixed(2);
                    
                    // Visual feedback
                    var timeDisplay = document.querySelector('.time-display');
                    timeDisplay.classList.add('updated');
                    setTimeout(function() {{
                        timeDisplay.classList.remove('updated');
                    }}, 1000);
                    
                    document.getElementById('status').textContent = `Start time set to ${{startTime.toFixed(2)}}s`;
                    
                    setTimeout(function() {{
                        document.getElementById('status').textContent = 'Start updated! Set end time or start looping';
                    }}, 2000);
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
                            document.getElementById('status').textContent = 'Loop updated! Ready to practice';
                        }}, 2000);
                    }} else {{
                        document.getElementById('status').textContent = 'End time must be after start time!';
                    }}
                }}
            }}

            function saveCurrentLoop() {{
                var loopName = document.getElementById('loopName').value.trim();
                if (!loopName) {{
                    loopName = `Loop ${{loopCounter}}`;
                }}
                
                if (endTime > startTime) {{
                    var loop = {{
                        name: loopName,
                        start: startTime,
                        end: endTime,
                        duration: endTime - startTime
                    }};
                    
                    savedLoops.push(loop);
                    loopCounter++;
                    
                    // Update the input for next loop
                    document.getElementById('loopName').value = `Loop ${{loopCounter}}`;
                    
                    // Show success message
                    var saveStatus = document.getElementById('saveStatus');
                    saveStatus.textContent = `✅ Saved: ${{loopName}}`;
                    setTimeout(function() {{
                        saveStatus.textContent = '';
                    }}, 3000);
                    
                    // Refresh the display
                    displaySavedLoops();
                }} else {{
                    document.getElementById('saveStatus').textContent = '❌ End time must be greater than start time!';
                    document.getElementById('saveStatus').style.color = 'red';
                    setTimeout(function() {{
                        document.getElementById('saveStatus').textContent = '';
                        document.getElementById('saveStatus').style.color = 'green';
                    }}, 3000);
                }}
            }}

            function loadLoop(index) {{
                var loop = savedLoops[index];
                if (loop) {{
                    startTime = loop.start;
                    endTime = loop.end;
                    
                    // Update display
                    document.getElementById('loopStart').textContent = startTime.toFixed(2);
                    document.getElementById('loopEnd').textContent = endTime.toFixed(2);
                    
                    // Visual feedback
                    var timeDisplay = document.querySelector('.time-display');
                    timeDisplay.classList.add('updated');
                    setTimeout(function() {{
                        timeDisplay.classList.remove('updated');
                    }}, 1000);
                    
                    // Jump to start of loaded loop
                    jumpToStart();
                    
                    document.getElementById('status').textContent = `Loaded: ${{loop.name}}`;
                    setTimeout(function() {{
                        document.getElementById('status').textContent = 'Loop loaded! Ready to practice';
                    }}, 2000);
                }}
            }}

            function deleteLoop(index) {{
                if (confirm(`Delete "${{savedLoops[index].name}}"?`)) {{
                    savedLoops.splice(index, 1);
                    displaySavedLoops();
                }}
            }}

            function displaySavedLoops() {{
                var container = document.getElementById('savedLoops');
                if (savedLoops.length === 0) {{
                    container.innerHTML = '<p style="color: #666; font-style: italic;">No saved loops yet. Set your loop times and save them!</p>';
                    return;
                }}
                
                var html = '';
                for (var i = 0; i < savedLoops.length; i++) {{
                    var loop = savedLoops[i];
                    html += `
                        <div class="loop-item">
                            <div class="loop-info">
                                <strong>${{loop.name}}</strong> - ${{loop.start.toFixed(2)}}s to ${{loop.end.toFixed(2)}}s (${{loop.duration.toFixed(2)}}s)
                            </div>
                            <div class="loop-actions">
                                <button class="load-button" onclick="loadLoop(${{i}})">🔄 Load</button>
                                <button class="delete-button" onclick="deleteLoop(${{i}})">🗑️ Delete</button>
                            </div>
                        </div>
                    `;
                }}
                container.innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """
    
    return html_code

def main():
    st.set_page_config(
        page_title="LoopLap by Jack Werner", 
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 LoopLap by Jack Werner")
    st.markdown("Perfect for practicing music sections with precise timing!")
    
    # Initialize session state
    if 'last_start_time' not in st.session_state:
        st.session_state.last_start_time = 0.0
    if 'default_duration' not in st.session_state:
        st.session_state.default_duration = 10.0
    if 'auto_end_time' not in st.session_state:
        st.session_state.auto_end_time = 10.0
    if 'example_url' not in st.session_state:
        st.session_state.example_url = ""
    if 'current_video_url' not in st.session_state:
        st.session_state.current_video_url = ""
    
    # Input section
    youtube_url = st.text_input(
        "YouTube URL:", 
        value=st.session_state.example_url or st.session_state.current_video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube URL here"
    )
    
    # Store the URL in session state when it changes
    if youtube_url and youtube_url != st.session_state.current_video_url:
        st.session_state.current_video_url = youtube_url
    
    # Clear the example URL after it's been used
    if youtube_url and st.session_state.example_url:
        st.session_state.example_url = ""
    
    # Use the stored URL as the source of truth
    active_url = st.session_state.current_video_url
    
    if active_url:
        video_id = extract_video_id(active_url)
        
        if video_id:
            st.success(f"✅ Video ID extracted: {video_id}")
            
            # Default times - user can adjust with video player controls
            start_time_total = 0.0
            end_time_total = 10.0
            
            st.markdown("### 🎬 Video Player with Loop Saving")
            
            # Create and display the player
            player_html = create_youtube_player(video_id, start_time_total, end_time_total)
            components.html(player_html, height=800)
            
            st.markdown("### 🎮 How to Use:")
            st.markdown("""
            - **🎯 Click timeline** to set start time
            - **▶️ Play/Pause**: Basic video controls  
            - **🔄 Loop ON/OFF**: Toggle automatic looping
            - **⏭️ Jump to Start**: Go to loop start time
            - **🔄 Reset**: Jump to start and play
            - **📍 Set Current as Start**: Use current position as start time
            - **🏁 Set Current as End**: Use current position as end time
            - **💾 Save Loop**: Save current loop times with a name (no page reload!)
            - **🔄 Load**: Load a saved loop and jump to it
            - **🗑️ Delete**: Remove saved loops
            
            **Pro tip**: The video keeps playing while you save/load loops!
            """)
                
        else:
            st.error("❌ Invalid YouTube URL. Please check the format.")
    
    else:
        st.info("👆 Enter a YouTube URL above to get started!")
        
        # Example YouTube links
        st.markdown("### 🎥 Try These Examples:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎸 Sugaree (Noblesville '21) - Dead & Co", key="example_1"):
                st.session_state.example_url = "https://www.youtube.com/watch?v=ZB8VdpXM1wk"
                st.rerun()
        
        with col2:
            if st.button("🎵 Bertha (Veneta '72) - Grateful Dead", key="example_2"):
                st.session_state.example_url = "https://www.youtube.com/watch?v=yTR8LDOc-rQ"
                st.rerun()
        
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
