$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function (event) {
        //var form_data = new FormData($('#upload-file')[0]);
        event.preventDefault();
//         var form_data = new FormData($('#upload-audio')[0], $('#upload-image')[0]);

        var form_data1 = new FormData($('#upload-audio')[0]);
        var form_data2 = new FormData($('#upload-image')[0]);
        var formValues1 = form_data1.entries()
        var formValues2 = form_data2.entries()
        
        var form_data = new FormData();
        while (!(ent = formValues1.next()).done) {
            // Note the change here 
            console.log(ent)
            form_data.append(`${ent.value[0]}[]`, ent.value[1])
        }
        console.log(form_data)
        while (!(ent = formValues2.next()).done) {
            // Note the change here 
            console.log(ent)
            form_data.append(`${ent.value[0]}[]`, ent.value[1])
        }
        console.log(form_data)
//         var form_data = new FormData();
//         form_data.append('audio_file', form_data1);
//         form_data.append('image_file', form_data2);
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
//             enctype: 'multipart/form-data',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Diagnosis result:  ' + data);
                console.log('Success!');
            },
        });
    });

});
