using NUnit.Framework;

namespace TinyLeastSquares.Tests;

/// <summary>
/// Minimal base class providing a WriteLine that routes to the NUnit test output.
/// </summary>
public abstract class TestFixtureBase
{
    protected static void WriteLine(string message) => TestContext.Out.WriteLine(message);
}
