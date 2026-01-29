const API_BASE = 'http://localhost:5508/api';

let videoData = null;
let excelData = null;
let nameMapping = {};

// Video Upload
document.getElementById('videoInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('videoFileName').textContent = file.name;
    const statusDiv = document.getElementById('videoStatus');
    statusDiv.className = 'status info';
    statusDiv.textContent = 'Uploading and processing video...';

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch(`${API_BASE}/upload-video`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            // Handle error response
            statusDiv.className = 'status error';
            statusDiv.textContent = `Error: ${data.detail || data.error || 'Unknown error occurred'}`;
            return;
        }

        if (data.success) {
            videoData = data;
            statusDiv.className = 'status success';
            statusDiv.textContent = `✓ Video processed! Transcript extracted with ${data.potential_names.length} potential names.`;
            
            // Show transcript section
            displayTranscript(data.transcript, data.potential_names);
        } else {
            statusDiv.className = 'status error';
            statusDiv.textContent = `Error: ${data.error || 'Processing failed'}`;
        }
    } catch (error) {
        statusDiv.className = 'status error';
        statusDiv.textContent = `Error: ${error.message}`;
        console.error('Upload error:', error);
    }
});

// Excel Upload
document.getElementById('excelInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('excelFileName').textContent = file.name;
    const statusDiv = document.getElementById('excelStatus');
    statusDiv.className = 'status info';
    statusDiv.textContent = 'Uploading and processing Excel file...';

    const formData = new FormData();
    formData.append('excel', file);

    try {
        const response = await fetch(`${API_BASE}/upload-excel`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            excelData = data;
            statusDiv.className = 'status success';
            statusDiv.textContent = `✓ Excel processed! Found ${data.count} names.`;
            
            // Update name mapping if transcript is already shown
            if (videoData) {
                updateNameMapping();
                // Show generate section
                document.getElementById('generateSection').style.display = 'block';
            }
        } else {
            statusDiv.className = 'status error';
            statusDiv.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        statusDiv.className = 'status error';
        statusDiv.textContent = `Error: ${error.message}`;
    }
});

function displayTranscript(transcript, potentialNames) {
    const transcriptSection = document.getElementById('transcriptSection');
    const transcriptText = document.getElementById('transcriptText');
    
    // Highlight potential names in transcript
    let highlightedTranscript = transcript;
    potentialNames.forEach(name => {
        // Escape special regex characters for Hindi/Unicode
        const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(escapedName, 'gi');
        highlightedTranscript = highlightedTranscript.replace(regex, `<span class="highlight">${name}</span>`);
    });
    
    transcriptText.innerHTML = highlightedTranscript;
    transcriptSection.style.display = 'block';
    
    // Add manual name input option
    addManualNameInput();
    
    // Create name mapping UI
    updateNameMapping();
}

function addManualNameInput() {
    const container = document.getElementById('nameMappingContainer');
    if (!container) return;
    
    // Check if input already exists
    if (document.getElementById('manualNameInput')) return;
    
    const manualInputDiv = document.createElement('div');
    manualInputDiv.style.marginBottom = '20px';
    manualInputDiv.style.padding = '15px';
    manualInputDiv.style.background = '#e7f3ff';
    manualInputDiv.style.borderRadius = '8px';
    manualInputDiv.style.border = '2px solid #667eea';
    
    manualInputDiv.innerHTML = `
        <h4 style="color: #667eea; margin-bottom: 10px;">➕ Add Name Manually</h4>
        <p style="color: #666; margin-bottom: 10px; font-size: 0.9em;">
            If a name is not detected automatically, type it here (e.g., "देवांग्लारिच"):
        </p>
        <div style="display: flex; gap: 10px;">
            <input 
                type="text" 
                id="manualNameInput" 
                placeholder="Enter name from transcript..."
                style="flex: 1; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;"
            />
            <button 
                onclick="addManualName()"
                style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;"
            >
                Add
            </button>
        </div>
    `;
    
    container.insertBefore(manualInputDiv, container.firstChild);
    
    // Add Enter key support
    document.getElementById('manualNameInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addManualName();
        }
    });
}

function addManualName() {
    const input = document.getElementById('manualNameInput');
    const name = input.value.trim();
    
    if (!name) {
        alert('Please enter a name');
        return;
    }
    
    // Check if name exists in transcript
    if (!videoData || !videoData.transcript.includes(name)) {
        const confirm = window.confirm(`"${name}" not found in transcript. Add anyway?`);
        if (!confirm) return;
    }
    
    // Add to potential names if not already there
    if (!videoData.potential_names.includes(name)) {
        videoData.potential_names.push(name);
    }
    
    // Clear input
    input.value = '';
    
    // Refresh the UI
    displayTranscript(videoData.transcript, videoData.potential_names);
}

function updateNameMapping() {
    if (!videoData) return;
    
    const container = document.getElementById('nameMappingContainer');
    if (!container) return;
    
    // Clear existing checkboxes (but keep manual input)
    const manualInputDiv = document.getElementById('manualNameInput')?.closest('div');
    container.innerHTML = '';
    if (manualInputDiv) {
        container.appendChild(manualInputDiv);
    }
    
    // Add instruction
    const instruction = document.createElement('p');
    instruction.style.marginBottom = '15px';
    instruction.style.color = '#666';
    instruction.textContent = excelData 
        ? 'Select which names from the video should be replaced. Each selected name will be replaced with Excel names to create personalized videos.'
        : 'Select which names from the video should be replaced. Upload Excel file to continue.';
    container.appendChild(instruction);
    
    // Add checkboxes for each potential name
    videoData.potential_names.forEach(name => {
        const item = document.createElement('div');
        item.className = 'name-mapping-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `name_${name}`;
        checkbox.value = name;
        checkbox.checked = nameMapping[name] === true; // Restore state
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                nameMapping[name] = true;
            } else {
                delete nameMapping[name];
            }
        });
        
        const label = document.createElement('label');
        label.htmlFor = `name_${name}`;
        label.textContent = `Replace "${name}" with Excel names`;
        
        item.appendChild(checkbox);
        item.appendChild(label);
        container.appendChild(item);
    });
    
    // Show generate section only if Excel is uploaded
    if (excelData) {
        document.getElementById('generateSection').style.display = 'block';
    }
}

async function generateVideos() {
    if (!videoData || !excelData || Object.keys(nameMapping).length === 0) {
        alert('Please select at least one name to replace!');
        return;
    }
    
    const generateBtn = document.getElementById('generateBtn');
    const statusDiv = document.getElementById('generateStatus');
    
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="loading"></span> Generating videos...';
    statusDiv.className = 'status info';
    statusDiv.textContent = 'Generating videos... This may take a few minutes.';
    
    try {
        const response = await fetch(`${API_BASE}/generate-videos`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_filename: videoData.filename,
                transcript: videoData.transcript,
                names_to_replace: Object.keys(nameMapping), // List of original names to replace
                excel_names: excelData.names,
                word_timestamps: videoData.word_timestamps || [], // Include word timestamps for precise replacement
                language: 'hi' // FORCE HINDI - Always Hindi, no exceptions
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status success';
            statusDiv.textContent = `✓ Successfully generated ${data.count} videos!`;
            
            // Show download section
            displayDownloads(data.videos);
        } else {
            statusDiv.className = 'status error';
            statusDiv.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        statusDiv.className = 'status error';
        statusDiv.textContent = `Error: ${error.message}`;
    } finally {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '🎥 Generate Videos';
    }
}

function displayDownloads(videos) {
    const downloadSection = document.getElementById('downloadSection');
    const downloadList = document.getElementById('downloadList');
    
    downloadList.innerHTML = '';
    
    videos.forEach(video => {
        const item = document.createElement('div');
        item.className = 'download-item';
        
        const title = document.createElement('h4');
        title.textContent = video;
        
        const btn = document.createElement('button');
        btn.className = 'download-btn';
        btn.textContent = '⬇ Download';
        btn.onclick = () => {
            window.location.href = `${API_BASE}/download/${video}`;
        };
        
        item.appendChild(title);
        item.appendChild(btn);
        downloadList.appendChild(item);
    });
    
    downloadSection.style.display = 'block';
}

