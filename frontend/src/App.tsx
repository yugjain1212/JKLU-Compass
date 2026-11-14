import { AsciiOcean } from "./ocean/AsciiOcean";
import { ChatHome } from "./components/ChatHome";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      {/* Full-screen animated ASCII ocean — light theme background */}
      <div className="ocean-layer">
        <AsciiOcean />
      </div>

      {/* UI sits above the ocean, pointer-events passthrough to canvas */}
      <div className="ui-layer">
        <ChatHome />
      </div>
    </div>
  );
}
