from ..main import application
from .helper import load_pipe


@application.route('/pipeline/<storage>/<name>/outer_fold/<outer_fold>/config/<config_index>/inner_fold/<inner_fold>/load')
def load_tested_config_for_inner_fold(storage, name, outer_fold, config_index, inner_fold):
    pipe = load_pipe(storage, name)

    config_index = int(config_index)
    outer_fold_index = int(int(outer_fold) - 1)
    inner_fold_index = int(int(inner_fold) - 1)

    inner_fold_object = pipe.outer_folds[outer_fold_index].tested_config_list[config_index].inner_folds[inner_fold_index]

    train_y_prediction = ',' . join(map(str, inner_fold_object.training.y_pred))
    train_x_prediction = ',' . join(map(str, list(range(1, len(inner_fold_object.training.y_pred)))))
    train_y_true       = ',' . join(map(str, inner_fold_object.training.y_true))
    train_x_true       = ',' . join(map(str, list(range(1, len(inner_fold_object.training.y_true)))))

    val_y_prediction = ','.join(map(str, inner_fold_object.training.y_pred))
    val_x_prediction = ','.join(map(str, list(range(1, len(inner_fold_object.training.y_pred)))))
    val_y_true = ','.join(map(str, inner_fold_object.training.y_true))
    val_x_true = ','.join(map(str, list(range(1, len(inner_fold_object.training.y_true)))))

    result = "<div class='tab-pane' id='config_" + str(config_index) + "_fold_" + str(inner_fold) + "'>"
    result += "<div class='row'>"

    ### START Training Graph ###

    result += "<div class='col-md-12'><div id='config_" + str(config_index) + "_fold_" + str(inner_fold) + "_training'></div>"
    result += "<script>"
    result += "var trace1 = {x: [" + train_x_true + "],"
    result += "y: [" + train_y_true + "],"
    result += " name: 'true',"
    result += " mode: 'markers'};"
    result += "var trace2 = {"
    result += "x: [" + train_x_prediction + "],"
    result += "y: [" + train_y_prediction + "],"
    result += " name: 'prediction',"
    result += " mode: 'markers'};"
    result += "var data = [ trace1, trace2 ];"
    result += "var layout = {title:'True/Predict for training set', width: 1600};"
    result += "Plotly.newPlot('config_" + str(config_index) + "_fold_" + str(inner_fold) + "_training', data, layout);"
    result += "</script></div>"

    ### START Validation Graph ###

    result += "<div class='col-md-12'><div id='config_" + str(config_index) + "_fold_" + str(inner_fold) + "_validation'></div>"
    result += "<script>"
    result += "var trace1 = {x: [" + val_x_true + "],"
    result += "y: [" + val_y_true + "],"
    result += " name: 'true',"
    result += " mode: 'markers'};"
    result += "var trace2 = {"
    result += "x: [" + val_x_prediction + "],"
    result += "y: [" + val_y_prediction + "],"
    result += " name: 'prediction',"
    result += " mode: 'markers'};"
    result += "var data = [ trace1, trace2 ];"
    result += "var layout = {title:'True/Predict for validation set', width: 1600};"
    result += "Plotly.newPlot('config_" + str(config_index) + "_fold_" + str(inner_fold) + "_validation', data, layout);"
    result += "</script></div>"
    result += "</div>"
    result += "</div>"

    return result


@application.route('/pipeline/<storage>/<name>/outer_fold/<outer_fold>/config/<config_index>/load')
def load_inner_fold_data_for_config(storage, name, outer_fold, config_index):
    pipe = load_pipe(storage, name)

    config_index = int(config_index)
    outer_fold_index = int(int(outer_fold) - 1)

    result = ""
    count = int(0)

    for inner_fold in pipe.outer_folds[outer_fold_index].tested_config_list[config_index].inner_folds:

        count += int(1)
        inner_fold_index = int(int(inner_fold.fold_nr) - 1)

        inner_fold_object = pipe.outer_folds[outer_fold_index].tested_config_list[config_index].inner_folds[inner_fold_index]

        train_y_prediction = ',' . join(map(str, inner_fold_object.training.y_pred))
        train_x_prediction = ',' . join(map(str, list(range(1, len(inner_fold_object.training.y_pred)))))
        train_y_true       = ',' . join(map(str, inner_fold_object.training.y_true))
        train_x_true       = ',' . join(map(str, list(range(1, len(inner_fold_object.training.y_true)))))

        val_y_prediction = ','.join(map(str, inner_fold_object.training.y_pred))
        val_x_prediction = ','.join(map(str, list(range(1, len(inner_fold_object.training.y_pred)))))
        val_y_true = ','.join(map(str, inner_fold_object.training.y_true))
        val_x_true = ','.join(map(str, list(range(1, len(inner_fold_object.training.y_true)))))

        if train_x_true or train_y_true or train_x_prediction or train_y_prediction or val_x_true or val_y_true or val_x_prediction or val_y_prediction:

            result += "<div class='tab-pane' id='config_" + str(config_index) + "_fold_" + str(inner_fold.fold_nr) + "'>"
            result += "<div class='row'>"

            ### START Training Graph ###

            result += "<div class='col-md-12'><div id='config_" + str(config_index) + "_fold_" + str(inner_fold.fold_nr) + "_training'></div>"
            result += "<script>"
            result += "var trace1 = {x: [" + train_x_true + "],"
            result += "y: [" + train_y_true + "],"
            result += " name: 'true',"
            result += " mode: 'markers'};"
            result += "var trace2 = {"
            result += "x: [" + train_x_prediction + "],"
            result += "y: [" + train_y_prediction + "],"
            result += " name: 'prediction',"
            result += " mode: 'markers'};"
            result += "var data = [ trace1, trace2 ];"
            result += "var layout = {title:'True/Predict for training set', width: 1600};"
            result += "Plotly.newPlot('config_" + str(config_index) + "_fold_" + str(inner_fold.fold_nr) + "_training', data, layout);"
            result += "</script></div>"

            ### START Validation Graph ###

            result += "<div class='col-md-12'><div id='config_" + str(config_index) + "_fold_" + str(inner_fold.fold_nr) + "_validation'></div>"
            result += "<script>"
            result += "var trace1 = {x: [" + val_x_true + "],"
            result += "y: [" + val_y_true + "],"
            result += " name: 'true',"
            result += " mode: 'markers'};"
            result += "var trace2 = {"
            result += "x: [" + val_x_prediction + "],"
            result += "y: [" + val_y_prediction + "],"
            result += " name: 'prediction',"
            result += " mode: 'markers'};"
            result += "var data = [ trace1, trace2 ];"
            result += "var layout = {title:'True/Predict for validation set', width: 1600};"
            result += "Plotly.newPlot('config_" + str(config_index) + "_fold_" + str(inner_fold.fold_nr) + "_validation', data, layout);"
            result += "</script></div>"
            result += "</div>"
            result += "</div>"

    return result
