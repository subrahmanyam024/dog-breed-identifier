const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const resultDiv = document.getElementById('result');
const loadingSpinner = document.getElementById('loading-spinner');

// Elements for user's uploaded image
const uploadedImageDisplay = document.getElementById('uploaded-image-display');
const imagePreview = document.getElementById('image-preview'); // Your uploaded image element
const uploadedImageLink = document.getElementById('uploaded-image-link');

// Removed: predictedImageDisplay, predictedImage, predictedImageLink references

// Event Listeners for Drag & Drop
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('border-blue-500', 'border-4');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('border-blue-500', 'border-4');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('border-blue-500', 'border-4');
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

// Event Listener for File Input Change
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    handleFile(file);
});

async function handleFile(file) {
    // Clear previous results and hide all dynamic elements
    resultDiv.textContent = '';
    resultDiv.classList.remove('text-red-500');
    loadingSpinner.classList.add('hidden');
    uploadedImageDisplay.classList.add('hidden'); // Hide uploaded image display initially
    uploadedImageLink.classList.add('hidden');


    if (file && file.type.startsWith('image/')) {
        // Display uploaded image preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadedImageLink.href = e.target.result; // Set link for uploaded image
            uploadedImageDisplay.classList.remove('hidden'); // Show the uploaded image display
            uploadedImageLink.classList.remove('hidden'); // Show the link for uploaded image
        };
        reader.readAsDataURL(file);

        // Show loading spinner while processing
        loadingSpinner.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData,
            });

            // Hide loading spinner once response is received
            loadingSpinner.classList.add('hidden');

            if (!response.ok) {
                const error = await response.json();
                resultDiv.textContent = `Error: ${error.error || 'Failed to upload image'}`;
                resultDiv.classList.add('text-red-500');
                return;
            }

            const data = await response.json();

            // Display prediction result
            resultDiv.textContent = `Predicted Breed: ${data.breed} (Confidence: ${data.confidence.toFixed(2)})`;

            // Removed: Logic for displaying predicted breed image

        } catch (error) {
            console.error('Error sending image to backend:', error);
            loadingSpinner.classList.add('hidden');
            resultDiv.textContent = 'Failed to communicate with the server.';
            resultDiv.classList.add('text-red-500');
        }
    } else {
        resultDiv.textContent = 'Please upload an image file.';
        resultDiv.classList.add('text-red-500');
        // Hide uploaded image display if file type is wrong
        uploadedImageDisplay.classList.add('hidden');
    }
}