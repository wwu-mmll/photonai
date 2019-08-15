from sklearn.metrics import confusion_matrix


def plotly_confusion_matrix(name, y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
    except:
        return ''

    trace = """var trace1 = {{
    type: 'heatmap', 
    x: ['Class 1', 'Class 2'], 
    y: ['Class 2', 'Class 1'], 
    zmin: '1', 
    z: [['{}', '{}'], ['{}', '{}']], 
    colorscale: [['0', 'rgb(255,245,240)'], ['0.2', 'rgb(254,224,210)'], ['0.4', 'rgb(252,187,161)'], ['0.5', 'rgb(252,146,114)'], ['0.6', 'rgb(251,106,74)'], ['0.7', 'rgb(239,59,44)'], ['0.8', 'rgb(203,24,29)'], ['0.9', 'rgb(165,15,21)'], ['1', 'rgb(103,0,13)']], 
    autocolorscale: false}};""".format(cm[1, 0], cm[1, 1], cm[0, 0], cm[0, 1])

    plot = """
var data = [trace1];
var layout = {{
  title: 'Confusion Matrix', 
  width: '400', 
  xaxis: {{
    title: 'Predicted value', 
    titlefont: {{
      size: '18', 
      color: '7f7f7f'
    }}
  }}, 
  yaxis: {{
    title: 'True Value', 
    titlefont: {{
      size: '18', 
      color: '7f7f7f'
    }}
  }}, 
}};
Plotly.newPlot('{}', data, layout);""".format(name)
    final_plot = trace + plot
    return final_plot
