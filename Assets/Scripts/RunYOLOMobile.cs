using System;
using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

/*
 *  YOLO Inference Script (Mobile-Optimized)
 *  ========================================
 *
 *  • Works with WebCamTexture on mobile and desktop.
 *  • Uses CPU backend on Android/iOS to avoid GPU buffer overflow.
 *  • Reduces model input size to 320×320 for mobile performance.
 *  • Limits inference to 5 FPS to prevent overheating.
 *
 */

public class RunYOLOMobile : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here")]
    public TextAsset classesAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    [Tooltip("Intersection over union threshold for NMS")]
    [Range(0, 1)] public float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold for NMS")]
    [Range(0, 1)] public float scoreThreshold = 0.5f;

#if UNITY_ANDROID || UNITY_IOS
    const BackendType backend = BackendType.CPU;
#else
    const BackendType backend = BackendType.GPUCompute;
#endif

    // Model input size (smaller = faster and safer on mobile)
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    private Worker worker;
    private string[] labels;
    private RenderTexture targetRT;
    private Sprite borderSprite;
    private WebCamTexture webcamTexture;
    private Tensor<float> centersToCorners;

    private Transform displayLocation;
    private List<GameObject> boxPool = new();

    // Inference timing
    private float inferenceInterval = 0.2f; // 5 FPS
    private float lastInferenceTime = 0f;

    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

    void Start()
    {
        Application.targetFrameRate = 30;

        // Load class labels
        labels = classesAsset.text.Split('\n');

        // Load YOLO model
        LoadModel();

        // Create render target
        targetRT = new RenderTexture(imageWidth, imageHeight, 0);
        displayLocation = displayImage.transform;

        // Initialize webcam
        if (WebCamTexture.devices.Length > 0)
        {
            WebCamDevice device = WebCamTexture.devices[0];
            webcamTexture = new WebCamTexture(device.name, imageWidth, imageHeight);
            webcamTexture.Play();
            displayImage.texture = webcamTexture;
        }
        else
        {
            Debug.LogError("No webcam found!");
            enabled = false;
            return;
        }

        // Create border sprite
        borderSprite = Sprite.Create(borderTexture,
            new Rect(0, 0, borderTexture.width, borderTexture.height),
            new Vector2(0.5f, 0.5f));
    }

    void LoadModel()
    {
        var model = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
            1, 0, 1, 0,
            0, 1, 0, 1,
            -0.5f, 0, 0.5f, 0,
            0, -0.5f, 0, 0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var modelOutput = Functional.Forward(model, inputs)[0];

        // YOLOv8 post-processing pipeline
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
        var allScores = modelOutput[0, 4.., ..];
        var scores = Functional.ReduceMax(allScores, 0);
        var classIDs = Functional.ArgMax(allScores, 0);
        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);
        var coords = Functional.IndexSelect(boxCoords, 0, indices);
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);

        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    void Update()
    {
        // Limit inference rate for mobile
        if (Time.time - lastInferenceTime >= inferenceInterval)
        {
            ExecuteML();
            lastInferenceTime = Time.time;
        }

        if (Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();
    }

    void ExecuteML()
    {
        ClearAnnotations();

        if (webcamTexture == null || !webcamTexture.didUpdateThisFrame)
            return;

        // Copy webcam frame to RenderTexture
        Graphics.Blit(webcamTexture, targetRT);

        using var inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);

        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        using var labelIDs = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        int boxesFound = output.shape[0];
        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            var box = new BoundingBox
            {
                centerX = output[n, 0] * scaleX - displayWidth / 2,
                centerY = output[n, 1] * scaleY - displayHeight / 2,
                width = output[n, 2] * scaleX,
                height = output[n, 3] * scaleY,
                label = labels[labelIDs[n]],
            };
            DrawBox(box, n, displayHeight * 0.05f);
        }
    }

    void DrawBox(BoundingBox box, int id, float fontSize)
    {
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }

        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        var label = panel.GetComponentInChildren<Text>();
        label.text = box.label;
        label.fontSize = (int)fontSize;
    }

    GameObject CreateNewBox(Color color)
    {
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.anchorMin = Vector2.zero;
        rt2.anchorMax = Vector2.one;
        rt2.offsetMin = new Vector2(10, 0);
        rt2.offsetMax = new Vector2(-10, 30);

        boxPool.Add(panel);
        return panel;
    }

    void ClearAnnotations()
    {
        foreach (var box in boxPool)
            box.SetActive(false);
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
        webcamTexture?.Stop();
    }
}
