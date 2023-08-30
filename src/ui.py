
import gradio as gr


def main(temp1, input, CPU_Core_Preprocess,CPU_Core_Cluster, Batch_Size, cluster_alg,hd_d, hd_q, checkbox, cluster_chrg, device, extra):
    return input



demo = gr.Interface(
    fn=main,
    inputs=[
        gr.Button("Default",variant="primary",size="sm"),
        gr.Dropdown(["PXD001468", "PCD00"]),
        gr.Slider(1,32, label="CPU Cores for Preprocessing"),
        gr.Slider(1,32, label="CPU Cores for Clustering"), 
        gr.Slider(0,5000, label="Batch Size"),
        gr.Radio(["dbscan", "hc_single", "hc_complete","hc_average"], label="Clustering Algorithm", info="Default is hc_complete"), 
        gr.Radio(["1024","2048","4096"], label= "HD Dimensions"), 
        gr.Radio(["4", "8", "16", "32", "64"], label= "HD Quantization level"),
        gr.CheckboxGroup(["Enabled"], label="GPU Clustering"),
        gr.CheckboxGroup(["1","2","3", "4", "5"], label="Cluster Charges"),
        #gr.Radio(["1","2","3", "4", "5"])
        #gr.Radio(["1","2","3", "4", "5"])
        gr.Slider(0,1, label='Threshold eps value for clustering'),
        gr.Radio(["CPU","GPU"], label="Hardware")
        ],
    outputs="text",
)
if __name__ == "__main__":
    demo.launch()

