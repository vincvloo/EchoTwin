// Which step the user is on, and what to call it. Pure functions of the server's `state` message:
// the dashboard and the phone page both use them, and tests/robot/test_ui_stage.py runs them under node.
(function (root) {
  const STEPS = {
    connecting: { n: 0, title: 'Connecting' },
    scan: { n: 1, title: 'Scan your table' },
    scanning: { n: 1, title: 'Building the twin' },
    ask: { n: 2, title: 'Say what to move' },
    watch: { n: 3, title: 'Watching it work' },
    teach: { n: 4, title: 'Show it how' },
    review: { n: 4, title: 'Keep this demo?' },
    stopped: { n: 0, title: 'Stopped' },
  };

  // s: the last `state` message (or null). o.scanning: a scan is being built right now.
  function stageOf(s, o) {
    o = o || {};
    if (!s) return 'connecting';
    if (s.halted) return 'stopped';
    if (o.scanning) return 'scanning';
    if (s.mode === 'review') return 'review';
    if (s.mode === 'teach') return 'teach';
    if (s.mode === 'move' || s.mode === 'replay' || s.mode === 'practice') return 'watch';
    return (s.props && s.props.length) ? 'ask' : 'scan';
  }

  // The one-line status shown over the video and in the phone header.
  function badge(s) {
    if (!s) return { text: 'Connecting…', tone: 'idle' };
    const name = s.name || 'Pip', t = s.task;
    if (s.halted) return { text: 'Stopped (' + s.halt_reason + ')', tone: 'stop' };
    switch (s.mode) {
      case 'move': return { text: name + ' is moving the ' + (t ? t.name : 'object') + (s.attempt > 1 ? ' · try ' + s.attempt : ''), tone: 'busy' };
      case 'practice': return { text: 'Practising in the twin', tone: 'busy' };
      case 'teach': return { text: s.recording ? 'Recording your demo' : 'Your turn: show me', tone: 'teach' };
      case 'replay': return { text: 'Replaying your video', tone: 'teach' };
      case 'review': return { text: 'Demo scored ' + (s.review ? s.review.score : '?'), tone: 'teach' };
    }
    return { text: s.authority === 'robot' ? name + ' has control' : 'Ready', tone: 'idle' };
  }

  // Why the robot asks to be shown (the `reason` of a `decision` message), in plain words.
  const REASONS = {
    new_kind: "I haven't moved anything this size yet",
    unsure: 'This layout is unlike what I have seen',
    video: 'Learning from your video',
    failed: 'The last try went wrong',
  };
  const reasonText = r => REASONS[r] || r || '';

  function sureLabel(pct) {
    return pct > 65 ? 'confident' : pct > 45 ? 'fairly sure' : 'not sure';
  }

  const api = { STEPS, stageOf, badge, reasonText, sureLabel };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Stage = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
