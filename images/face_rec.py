import numpy

import face_recognition

import PIL
import PIL.ImageDraw
import PIL.ImageColor

make_plots = False

def features(imagefile):
    image = face_recognition.load_image_file(imagefile)
    pil_image = PIL.Image.fromarray(image)
    width, height = pil_image.size

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    print("{}: {} faces".format(imagefile,len(face_locations)))
    for face_loc in face_locations:
        (top, right, bottom, left) = face_loc
        if make_plots:
            # PIL wants [x0, y0, x1, y1], where x1 >= x0 and y1 >= y0
            rect = [left,top,right,bottom]
            draw = PIL.ImageDraw.Draw(pil_image)
            draw.rectangle(rect, outline=PIL.ImageColor.colormap["red"], width=1)
        rel_size = (top-bottom)*(left-right)/(width*height)
        print("  {}".format(rel_size))

    face_locations = face_recognition.face_locations(image)
    print("{}: {} faces".format(imagefile,len(face_locations)))
    for face_loc in face_locations:
        (top, right, bottom, left) = face_loc
        if make_plots:
            # PIL wants [x0, y0, x1, y1], where x1 >= x0 and y1 >= y0
            rect = [left,top,right,bottom]
            draw = PIL.ImageDraw.Draw(pil_image)
            draw.rectangle(rect, outline=PIL.ImageColor.colormap["green"], width=1)
        rel_size = (bottom-top)*(right-left)/(width*height)
        print("  {}".format(rel_size))

    face_landmarks = face_recognition.face_landmarks(image)
    for face_landmark in face_landmarks:
        face_landmark_points = {}
        for k in face_landmark.keys():
            points = numpy.array(face_landmark[k])
            x = points[:,0].mean()
            y = points[:,1].mean()
            face_landmark_points[k] = (x,y)
            if make_plots:
                draw = PIL.ImageDraw.Draw(pil_image)
                draw.text((x,y), k, fill="yellow", anchor="ms")
        features = {}
        # Eyes are eyebrows are expected to be at almost the same height
        # If any is missing, give max relative error
        try:
            (x0,y0) = face_landmark_points["left_eyebrow"]
            (x1,y1) = face_landmark_points["right_eyebrow"]
            features["eyebrows"] = numpy.abs(y0-y1)/(bottom-top)
        except:
            features["eyebrows"] = 1.0
        try:
            (x0,y0) = face_landmark_points["left_eye"]
            (x1,y1) = face_landmark_points["right_eye"]
            features["eyes"] = numpy.abs(y0-y1)/(bottom-top)
        except:
            features["eyes"] = 1.0
        # nose_bridge/nose_tip and top_lip/bottom_lip are expected to be at almost the same width
        try:
            (x0,y0) = face_landmark_points["nose_bridge"]
            (x1,y1) = face_landmark_points["nose_tip"]
            features["nose"] = numpy.abs(x0-x1)/(right-left)
        except:
            features["nose"] = 1.0
        try:
            (x0,y0) = face_landmark_points["top_lip"]
            (x1,y1) = face_landmark_points["bottom_lip"]
            features["lips"] = numpy.abs(x0-x1)/(right-left)
        except:
            features["lips"] = 1.0
        print(features)

    if make_plots:
        pil_image.save(imagefile+"_FACE.png","PNG")
