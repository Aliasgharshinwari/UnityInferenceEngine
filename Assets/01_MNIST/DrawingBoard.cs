using UnityEngine;
using UnityEngine.EventSystems;

public class DrawingBoard : MonoBehaviour
{
    public int width = 280;   // display size (10x scale for visibility)
    public int height = 280;
    public int modelInputSize = 28; // actual MNIST resolution
    public int brushSize = 12; // Default radius (can be changed via UI)

    private Texture2D drawTexture;
    private Color[] clearColors;
    private Color drawColor = Color.white;
    private Color backgroundColor = Color.black;
    public LayerMask drawingLayer;

    // 👇 Double-tap variables
    [Header("Double Tap Settings")]
    public float doubleTapTime = 0.3f; // Max delay between taps
    private float lastTapTime = 0f;


    void Start()
    {
        drawTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        clearColors = new Color[width * height];
        for (int i = 0; i < clearColors.Length; i++)
            clearColors[i] = backgroundColor;
        ClearBoard();

        // Assign the texture to a Quad or RawImage in your scene
        GetComponent<Renderer>().material.mainTexture = drawTexture;
    }

    void Update()
    {
        HandleMouseInput();
        HandleTouchInput();
    }

    // ===============================
    // 🖱️ Mouse Input (Desktop)
    // ===============================
    void HandleMouseInput()
    {
        if (Input.GetMouseButton(0))
        {
            if (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject())
                return; // Skip if clicking on UI

            Vector3 mousePos = Input.mousePosition;
            Ray ray = Camera.main.ScreenPointToRay(mousePos);
            if (Physics.Raycast(ray, out RaycastHit hit, Mathf.Infinity, drawingLayer))
            {
                DrawAt(hit.textureCoord);
            }
        }

        // 🧽 Double-click to clear
        if (Input.GetMouseButtonDown(0))
        {
            if (Time.time - lastTapTime < doubleTapTime)
            {
                Debug.Log("🧽 Double-click detected — clearing board!");
                ClearBoard();
                lastTapTime = 0f;
            }
            else
            {
                lastTapTime = Time.time;
            }
        }

        // Optional: manual clear via key
        if (Input.GetKeyDown(KeyCode.C))
            ClearBoard();
    }

    // ===============================
    // 🖐️ Touch Input (Mobile)
    // ===============================
    void HandleTouchInput()
    {
        if (Input.touchCount > 0)
        {
            foreach (Touch touch in Input.touches)
            {
                if (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject(touch.fingerId))
                    continue; // Skip if touching UI

                // Draw while moving or holding finger
                if (touch.phase == TouchPhase.Moved || touch.phase == TouchPhase.Stationary)
                {
                    Vector3 touchPos = touch.position;
                    Ray ray = Camera.main.ScreenPointToRay(touchPos);
                    if (Physics.Raycast(ray, out RaycastHit hit, Mathf.Infinity, drawingLayer))
                    {
                        DrawAt(hit.textureCoord);
                    }
                }

                // 🧽 Double-tap detection
                if (touch.phase == TouchPhase.Ended)
                {
                    if (Time.time - lastTapTime < doubleTapTime)
                    {
                        Debug.Log("🧽 Double-tap detected — clearing board!");
                        ClearBoard();
                        lastTapTime = 0f;
                    }
                    else
                    {
                        lastTapTime = Time.time;
                    }
                }
            }
        }
    }

    // ===============================
    // 🖌️ Draw Function
    // ===============================
    void DrawAt(Vector2 texCoord)
    {
        int x = (int)(texCoord.x * width);
        int y = (int)(texCoord.y * height);


        int brushRadius = Mathf.RoundToInt(brushSize);


        for (int i = -brushSize; i <= brushSize; i++)
        {
            for (int j = -brushSize; j <= brushSize; j++)
            {
                int px = Mathf.Clamp(x + i, 0, width - 1);
                int py = Mathf.Clamp(y + j, 0, height - 1);

                // Distance from center of brush
                float distance = Mathf.Sqrt(i * i + j * j);

                // Compute alpha falloff (soft edge)
                if (distance < brushRadius)
                {
                    float alpha = Mathf.Exp(-4 * (distance / brushRadius) * (distance / brushRadius)); // Gaussian falloff

                    Color current = drawTexture.GetPixel(px, py);
                    Color blended = Color.Lerp(current, drawColor, alpha);

                    drawTexture.SetPixel(px, py, blended);
                }
            }
        }

        drawTexture.Apply();
    }


    // ===============================
    // 🧽 Clear Function
    // ===============================
    public void ClearBoard()
    {
        drawTexture.SetPixels(clearColors);
        drawTexture.Apply();
    }

    // ===============================
    // 🔄 Resize for Model
    // ===============================
    public Texture2D GetResizedForModel()
    {
        Texture2D resized = new Texture2D(modelInputSize, modelInputSize, TextureFormat.RGBA32, false);
        Color[] smallPixels = new Color[modelInputSize * modelInputSize];

        for (int y = 0; y < modelInputSize; y++)
        {
            for (int x = 0; x < modelInputSize; x++)
            {
                float gx = (float)x / (modelInputSize - 1);
                float gy = (float)y / (modelInputSize - 1);
                smallPixels[y * modelInputSize + x] = drawTexture.GetPixelBilinear(gx, gy);
            }
        }

        resized.SetPixels(smallPixels);
        resized.Apply();
        return resized;
    }

    public void ChangeRadius(float value)
    {
        brushSize = (int)value;
    }

}
