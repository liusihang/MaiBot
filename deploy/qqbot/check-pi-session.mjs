// Read-only acceptance of a migrated Pi session. Input contains private config;
// output contains only session identity and verification results.
import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

process.stdin.setEncoding('utf8');
let input='';for await(const chunk of process.stdin)input+=chunk;
const {file,contract,config}=JSON.parse(input);
for(const tool of contract.tools) {
  const binding=contract.toolBindings[tool.name];
  if(binding?.definition) {
    assert.deepEqual(binding.definition,{
      name:tool.name,description:tool.description,parameters_schema:tool.parameters,
    },`Tool declaration and execution binding differ: ${tool.name}`);
  }
}
const root=process.env.MAIBOT_ROOT??process.cwd();
const {PiRuntime}=await import(pathToFileURL(resolve(root,'runtime/pi_agent/runtime.mjs')).href);
const runtime=new PiRuntime(async(method)=>{
  assert.equal(method,'event','Session acceptance cannot execute a business tool');return {};
});
try {
  const state=await runtime.handle('init',{...contract,sessionDir:dirname(file),config,requireDelivery:true});
  assert.equal(state.error,null);assert.deepEqual(state.unresolvedToolCalls,[]);assert.equal(state.isRunning,false);
  console.log(JSON.stringify({initialized:true,sessionId:contract.sessionId,toolCount:contract.tools.length,
    queuedMessageCount:state.queuedMessageCount,toolBindingsVerified:true,modelRequests:0,qqSends:0}));
} finally {await runtime.shutdown();}
