using UnityEngine;
using Unity.InferenceEngine;
using TMPro;

public class ModelExecutionMNISTCNN : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private DrawingBoard drawingBoard; 
    [SerializeField] private TMP_Text predictedDigitText;
    [SerializeField] private int predicted;

    private Worker m_Worker;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    void Update()
    {
        // Press SPACE to predict the drawn digit
        //if (Input.GetKeyDown(KeyCode.Space))
        //{
            Texture2D drawn = drawingBoard.GetResizedForModel();
            float[] inputData = TextureToMNISTInput(drawn);

            // Create input tensor
            var inputTensor = new Tensor<float>(new TensorShape(1, 28, 28), inputData);

            // Run inference
            m_Worker.Schedule(inputTensor);
            var outputGPU = m_Worker.PeekOutput() as Tensor<float>;
            var outputCPU = outputGPU.ReadbackAndClone();

            // ArgMax to get digit
            predicted = ArgMax(outputCPU);
        //Debug.Log($"🧮 Predicted Digit: {predicted}");
            predictedDigitText.text = predicted.ToString();
            inputTensor.Dispose();
            outputCPU.Dispose();
        //}
    }

    private float[] TextureToMNISTInput(Texture2D tex)
    {
        Color[] pixels = tex.GetPixels();
        float[] data = new float[28 * 28];
        for (int i = 0; i < pixels.Length; i++)
        {
            data[i] = pixels[i].grayscale; // already 0..1 range
        }
        return data;
    }

    private int ArgMax(Tensor<float> tensor)
    {
        int index = 0;
        float max = float.MinValue;
        for (int i = 0; i < tensor.shape.length; i++)
        {
            float val = tensor[i];
            if (val > max) { max = val; index = i; }
        }
        return index;
    }

    void OnDisable()
    {
        m_Worker?.Dispose();
    }
}
