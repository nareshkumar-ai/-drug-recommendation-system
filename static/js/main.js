$(document).ready(function () {
    let currentDiagnosedDisease = '';

    // Initialize Select2
    $('#symptoms').select2({
        placeholder: 'Search and select symptoms...',
        allowClear: true,
        closeOnSelect: false
    });

    // Chatbot Toggle Logic
    $('#chatToggle').on('click', function () {
        $('#chatWidget').addClass('active');
        $(this).fadeOut();
    });

    $('#closeChat').on('click', function () {
        $('#chatWidget').removeClass('active');
        $('#chatToggle').fadeIn();
    });

    // Prediction Form Submission
    $('#predictionForm').on('submit', function (e) {
        e.preventDefault();
        const selectedSymptoms = $('#symptoms').val();

        if (!selectedSymptoms || selectedSymptoms.length === 0) {
            alert('Please select at least one symptom.');
            return;
        }

        // UI State
        $('#predictBtn').prop('disabled', true);
        $('.btn-text').hide();
        $('#predictLoader').show();

        $.ajax({
            url: '/predict',
            method: 'POST',
            data: { symptoms: selectedSymptoms.join(',') },
            success: function (response) {
                displayResults(response);
            },
            error: function (err) {
                console.error(err);
                alert('An error occurred while predicting. Please try again.');
            },
            complete: function () {
                $('#predictBtn').prop('disabled', false);
                $('.btn-text').show();
                $('#predictLoader').hide();
            }
        });
    });

    function displayResults(data) {
        currentDiagnosedDisease = data.disease || '';
        const diseaseIcon = $('<i class="fas fa-notes-medical"></i> ');
        $('#diseaseName').empty().append(diseaseIcon).append(document.createTextNode(data.disease || ''));
        $('#diseaseDesc').text(data.description || 'No description available.');

        // Clear and fill lists
        fillTagList('#medsList', data.medications);
        fillTagList('#precList', data.precautions);
        fillTagList('#dietList', data.diets);
        fillTagList('#workoutList', data.workout);

        $('#resultCard').fadeIn();

        // Scroll to results
        $('html, body').animate({
            scrollTop: $("#resultCard").offset().top - 50
        }, 800);
    }

    function fillTagList(selector, items) {
        const container = $(selector);
        container.empty();
        if (items && Array.isArray(items) && items.length > 0) {
            let count = 0;
            items.forEach(item => {
                const text = (item !== null && item !== undefined) ? String(item).trim() : '';
                if (text) {
                    const tag = $('<span class="tag"></span>').text(text);
                    container.append(tag);
                    count++;
                }
            });
            if (count === 0) {
                container.append('<p>No specific recommendations found.</p>');
            }
        } else {
            container.append('<p>No specific recommendations found.</p>');
        }
    }

    // Chat Logic
    $('#sendChat').on('click', sendMessage);
    $('#chatInput').on('keypress', function (e) {
        if (e.which == 13) sendMessage();
    });

    function sendMessage() {
        const message = $('#chatInput').val().trim();
        if (!message) return;

        appendMessage('user', message);
        $('#chatInput').val('');

        // Loading message
        const $loadingMsg = $('<div class="message bot">...typing</div>');
        $('#chatMessages').append($loadingMsg);
        scrollToBottom();

        $.ajax({
            url: '/chat',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ 
                message: message,
                disease: currentDiagnosedDisease 
            }),
            success: function (response) {
                $loadingMsg.remove();
                appendMessage('bot', response.response);
            },
            error: function () {
                $loadingMsg.remove();
                appendMessage('bot', "I'm sorry, I encountered an error. Please try again.");
            }
        });
    }

    function appendMessage(role, text) {
        const $msg = $('<div class="message"></div>').addClass(role).text(text);
        $('#chatMessages').append($msg);
        scrollToBottom();
    }

    function scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
