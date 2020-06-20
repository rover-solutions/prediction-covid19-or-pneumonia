let select_image = false;
let conteiner_response = document.getElementById('response');

function chooseFile() {
    conteiner_response.innerHTML = '';
    $("#file-input").click();
}

function predict() {
    if (select_image) {
        conteiner_response.innerHTML = 'Loading...';
        let formData = new FormData($(".form-toolbar")[0]);
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            success: function (response) {
                if (response.error) {
                    alert(response.error);
                } else {
                    conteiner_response.innerHTML = (response.is_covid) ? 'Covid-19' : 'Pneumonia';
                    conteiner_response.innerHTML += ' - Percentage: ' + response.percentage + '%';
                }
            },
            contentType: false,
            processData: false,
            cache: false
        });
    } else {
        alert('Select an image before predicting!!');
    }   
}

window.addEventListener('load', function() {
    document.getElementById('file-input').addEventListener('change', function() {
        if (this.files && this.files[0]) {
            var img = document.querySelector('img');
            img.src = URL.createObjectURL(this.files[0]);
            document.getElementById('no-image').style.display = 'none';
            document.getElementById('image').style.display = 'block';
            select_image = true;
        }
    });
});