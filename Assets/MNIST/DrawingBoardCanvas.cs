using UnityEngine;
using UnityEngine.EventSystems;

public class DrawingBoardCanvas : MonoBehaviour
{
    public int width = 280;   // display size (10x scale for visibility)
    public int height = 280;
    public int modelInputSize = 28; // actual MNIST resolution

    private Texture2D drawTexture;
    private Color[] clearColors;
    private Color drawColor = Color.white;
    private Color backgroundColor = Color.black;

    [Range(1, 50)]
    public int brushSize = 20;  // adjustable brush size

    private void Start()
    {
        drawTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        clearColors = new Color[width * height];
        for (int i = 0; i < clearColors.Length; i++)
            clearColors[i] = backgroundColor;

        drawTexture.SetPixels(clearColors);
        drawTexture.Apply();

        GetComponent<UnityEngine.UI.RawImage>().texture = drawTexture;
    }

    private void Update()
    {
        if (Input.GetMouseButton(0) && !EventSystem.current.IsPointerOverGameObject())
        {
            Vector2 localPos;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                GetComponent<RectTransform>(),
                Input.mousePosition,
                null,
                out localPos
            );

            float px = (localPos.x + width / 2);
            float py = (localPos.y + height / 2);
            DrawCircle((int)px, (int)py);
        }
    }

    void DrawCircle(int cx, int cy)
    {
        for (int x = -brushSize; x <= brushSize; x++)
        {
            for (int y = -brushSize; y <= brushSize; y++)
            {
                if (x * x + y * y <= brushSize * brushSize)
                {
                    int px = cx + x;
                    int py = cy + y;
                    if (px >= 0 && px < width && py >= 0 && py < height)
                        drawTexture.SetPixel(px, py, drawColor);
                }
            }
        }
        drawTexture.Apply();
    }

    public void ClearBoard()
    {
        drawTexture.SetPixels(clearColors);
        drawTexture.Apply();
    }

    public Texture2D GetResizedForModel()
    {
        // Convert to 28x28 for model input
        Texture2D scaled = new Texture2D(modelInputSize, modelInputSize, TextureFormat.RGBA32, false);
        Graphics.ConvertTexture(drawTexture, scaled);
        return scaled;
    }
}
