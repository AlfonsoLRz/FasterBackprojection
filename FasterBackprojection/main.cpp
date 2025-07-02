#include "stdafx.h"

#include "Renderer.h"
#include "Window.h"

int main(int argc, char* argv)
{
    Window* window = Window::getInstance();
    Renderer* renderer = Renderer::getInstance();

    try
    {
        window->init("Ray Tracing");
        window->loop();
    }
    catch (const std::exception& exception)
    {
        std::cout << exception.what() << std::endl;
    }

    // - Una vez terminado el ciclo de eventos, liberar recursos, etc.
    std::cout << "Finishing application..." << '\n';

    // - Esta llamada es para impedir que la consola se cierre inmediatamente tras la
    // ejecución y poder leer los mensajes. Se puede usar también getChar();
    system("pause");

    return 0;
}