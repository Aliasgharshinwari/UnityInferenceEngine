using UnityEngine;
using Unity.InferenceEngine;
using TMPro;

public class ModelExecutionMNIST : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private DrawingBoard drawingBoard;
    [SerializeField] private TMP_Text predictedDigitText;
    [SerializeField] private int predicted;

    private Worker m_Worker;
    private bool isCNN; // Will be determined automatically
    private bool modelLoaded = false;


    void Start()
    {
        LoadModel(modelAsset);
    }

    public void LoadModel(ModelAsset newModelAsset)
    {
        // Dispose any existing worker first
        m_Worker?.Dispose();

        modelAsset = newModelAsset;
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
        modelLoaded = true;

        // Detect whether this model expects 2D or 1D input
        var inputShape = model.inputs[0].shape;
        // For CNN: (1, 28, 28) or (1, 1, 28, 28)
        // For MLP: (1, 784)
        isCNN = inputShape.rank >= 3;

        Debug.Log($"🧠 Model loaded dynamically. Type: {(isCNN ? "CNN (2D)" : "MLP (1D)")}");
    }


    void Update()
    {
        if (!modelLoaded) return;

        // Run inference only when the user is drawing
        if (drawingBoard.IsDrawing)
            RunInference();
    }

    private void RunInference()
    {
        Texture2D drawn = drawingBoard.GetResizedForModel();
        float[] inputData = TextureToMNISTInput(drawn);

        Tensor<float> inputTensor;

        if (isCNN)
            inputTensor = new Tensor<float>(new TensorShape(1, 28, 28), inputData);
        else
            inputTensor = new Tensor<float>(new TensorShape(1, 28 * 28), inputData);

        // Run inference
        m_Worker.Schedule(inputTensor);
        var outputGPU = m_Worker.PeekOutput() as Tensor<float>;
        var outputCPU = outputGPU.ReadbackAndClone();

        predicted = ArgMax(outputCPU);
        predictedDigitText.text = predicted.ToString();

        inputTensor.Dispose();
        outputCPU.Dispose();
    }

    private float[] TextureToMNISTInput(Texture2D tex)
    {
        Color[] pixels = tex.GetPixels();
        float[] data = new float[28 * 28];
        for (int i = 0; i < pixels.Length; i++)
            data[i] = pixels[i].grayscale;
        return data;
    }

    private int ArgMax(Tensor<float> tensor)
    {
        int index = 0;
        float max = float.MinValue;
        for (int i = 0; i < tensor.shape.length; i++)
        {
            float val = tensor[i];
            if (val > max)
            {
                max = val;
                index = i;
            }
        }
        return index;
    }

    void OnDisable()
    {
        m_Worker?.Dispose();
    }
}
