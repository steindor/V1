
// $(document).ready(function () {
//     $('img').click(function (e) {
//         var offset = $(this).offset();
//         alert(e.pageX - offset.left);
//         alert(e.pageY - offset.top);
//     });
// });

var canvas = document.getElementById("canvas"),
ctx = canvas.getContext("2d");

canvas.width = 800;
canvas.height = 800;

var background;


var background = new Image();
background.src = $('.next-image').val()

// Make sure the image is loaded first otherwise nothing will draw.
background.onload = function () {
    ctx.drawImage(background, 0, 0);
}


$(document).on("keyup", function(e){
    if(e.keyCode == 13){
        var $el = $('.rectangle')
        var ann_arr = []
        $el.each(function(){
            $bbox = $(this)
            var $pos = $bbox.position()
            var left = $pos.left
                , top = $pos.top
                , width = parseInt($bbox.css('width').replace("px", ""))
                , height = parseInt($bbox.css('height').replace("px", ""))
                , img_path = background.src

            var img_data = {
                left: left,
                top: top,
                width: width,
                height: height,
                path: img_path
            }

            ann_arr.push(img_data)
        });

        var $pos = $el.position()
        var left = $pos.left
        , top = $pos.top
        , width = parseInt($el.css('width').replace("px",""))
        , height = parseInt($el.css('height').replace("px",""))
        , img_path = background.src

        var img_data = {
            left: left,
            top: top,
            width: width,
            height: height,
            path: img_path
        }


        $.post("/save_bbox", { img_data: JSON.stringify(ann_arr)}, function(nextPhoto){
            background.src = nextPhoto
            $('.rectangle').remove()
        })    

    }
})

initDraw(document.getElementById('canvas'));

function initDraw(canvas) {
    var mouse = {
        x: 0,
        y: 0,
        startX: 0,
        startY: 0
    };
    function setMousePosition(e) {
        var ev = e || window.event; //Moz || IE
        if (ev.pageX) { //Moz
            mouse.x = ev.pageX + window.pageXOffset;
            mouse.y = ev.pageY + window.pageYOffset;
        } else if (ev.clientX) { //IE
            mouse.x = ev.clientX + document.body.scrollLeft;
            mouse.y = ev.clientY + document.body.scrollTop;
        }
    };

    var element = null;
    canvas.onmousemove = function (e) {
        setMousePosition(e);
        if (element !== null) {

            element.style.width = Math.abs(mouse.x - mouse.startX) + 'px';
            element.style.height = Math.abs(mouse.y - mouse.startY) + 'px';
            element.style.left = (mouse.x - mouse.startX < 0) ? mouse.x + 'px' : mouse.startX + 'px';
            element.style.top = (mouse.y - mouse.startY < 0) ? mouse.y + 'px' : mouse.startY + 'px';
        }
    }

    $(document).on("click", ".rectangle", function () {
        element = null
    })

    canvas.onclick = function (e) {
        if (element !== null) {
            element = null;
        } else {
            mouse.startX = mouse.x;
            mouse.startY = mouse.y;
            var offset = $(this).offset();
            element = document.createElement('div');
            element.className = 'rectangle'
            element.style.left = mouse.x  + 'px';
            element.style.top = mouse.y - offset.top + 'px';
            document.body.appendChild(element)
        }
    }
}